//! Frame-level multithreaded decoding.
//!
//! The macroblock decode loop is inherently serial within a picture
//! (CABAC neighbor dependencies), so parallelism comes from pipelining
//! *pictures*. The coordinator thread does all header-level work in coded
//! order — slice-header parsing, POC computation, reference-list
//! construction — against a "shadow" DPB of planned pictures whose pixels
//! have not been decoded yet. Worker threads run the per-picture MB decode
//! loops and deblocking, then commit results back in coded order.
//!
//! Because the shadow DPB replays exactly the same bookkeeping (insert,
//! sliding window, MMCO) with the same metadata in the same order as the
//! serial [`Decoder`](crate::decoder::Decoder), reference lists are
//! identical by construction and decoded output is bit-exact with the
//! serial path. The full test suite runs every multi-frame stream through
//! this decoder and compares byte-for-byte.
//!
//! Dependencies are tracked at picture granularity: a picture's MB decode
//! starts once all its reference pictures have been committed (decoded,
//! deblocked, and published). On typical B-frame content (e.g. x264
//! `bframes=3`) this exposes 3-5 pictures of parallelism; pure P-chains
//! reference their immediate predecessor and gain little — the same
//! dependency structure FFmpeg's frame-threaded H.264 decoder exploits.
//!
//! Limitations: field pictures (interlaced coded as separate fields) are
//! not supported and return an error — use the serial `Decoder` for
//! those. MBAFF frame pictures are fine.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

use crate::decoder::{
    apply_reference_marking, crop_to_display, deblock_picture, prepare_slice_job, run_slice_job,
    take_decoded_picture, Frame, PictureState, SliceJob, SliceJobShell,
};
use crate::dpb::{DecodedPicture, Dpb, PicRef, PictureStructure};
use crate::error::DecodeError;
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{parse_pps, Pps};
use crate::sps::{parse_sps, Sps};

/// Slot that receives the real decoded picture when its frame commits.
/// Workers read reference slots only after the commit barrier guarantees
/// they are filled.
pub(crate) type SharedSlot = Arc<OnceLock<Arc<DecodedPicture>>>;

/// A planned (not yet decoded) picture handle for the shadow DPB.
#[derive(Clone)]
pub(crate) struct PlannedPic {
    slot: SharedSlot,
    poc: i32,
    frame_num: u32,
    structure: PictureStructure,
}

impl PicRef for PlannedPic {
    fn poc(&self) -> i32 {
        self.poc
    }
    fn frame_num(&self) -> u32 {
        self.frame_num
    }
    fn structure(&self) -> PictureStructure {
        self.structure
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.slot, &other.slot)
    }
}

/// One accumulated slice of the currently-open picture.
struct ThreadSlice {
    shell: SliceJobShell<PlannedPic>,
    rbsp: Vec<u8>,
}

/// Metadata of the currently-open picture (assigned at its first slice).
struct OpenPicture {
    seq: u64,
    slot: SharedSlot,
}

struct CommitBarrier {
    /// Highest consecutively-committed seq + 1 (i.e. pictures 0..committed
    /// are complete and their slots are filled).
    committed: Mutex<u64>,
    cv: Condvar,
}

/// H.264 decoder that pipelines pictures across worker threads and emits
/// frames in display order — the threaded counterpart of
/// [`OrderedDecoder`](crate::decoder::OrderedDecoder).
///
/// ```no_run
/// use rust_h264::threading::ThreadedDecoder;
/// use rust_h264::nal::parse_annex_b;
///
/// let h264 = std::fs::read("input.h264").unwrap();
/// let nals = parse_annex_b(&h264);
/// let mut decoder = ThreadedDecoder::new(4);
/// for nal in &nals {
///     for frame in decoder.decode_nal(nal).unwrap() {
///         // frame is in display order, bit-identical to the serial decoder
///     }
/// }
/// for frame in decoder.flush() {
///     // final buffered frames
/// }
/// ```
pub struct ThreadedDecoder {
    threads: usize,
    sps_table: HashMap<u32, Sps>,
    pps_table: HashMap<u32, Pps>,
    /// Shadow DPB: same bookkeeping as the serial decoder, on planned
    /// pictures whose pixels arrive at commit time.
    dpb: Dpb<PlannedPic>,
    /// Picture currently accumulating slices (None between pictures).
    pending: Option<PictureState>,
    open_slices: Vec<ThreadSlice>,
    open: Option<OpenPicture>,
    next_seq: u64,
    /// First picture not yet committed; frames commit strictly in order.
    commit_next: u64,
    barrier: Arc<CommitBarrier>,
    /// Slot (and metadata) per scheduled seq, for commits.
    slots: HashMap<u64, (SharedSlot, u32)>, // (slot, gop_id)
    /// Slot identity -> seq, to compute a picture's commit dependencies.
    slot_seqs: HashMap<usize, u64>,
    inflight: VecDeque<JoinHandle<()>>,
    completed: BTreeMap<u64, Result<PictureState, DecodeError>>,
    done_rx: Receiver<(u64, Result<PictureState, DecodeError>)>,
    done_tx: Sender<(u64, Result<PictureState, DecodeError>)>,
    first_error: Option<DecodeError>,
    // Display-order reorder buffer (OrderedDecoder semantics).
    buffer: Vec<(u32, Frame)>,
    gop_id: u32,
    max_depth: usize,
    decoded_frames: AtomicU64,
}

impl ThreadedDecoder {
    /// Create a decoder using `threads` worker threads (clamped to >= 1).
    pub fn new(threads: usize) -> Self {
        let (done_tx, done_rx) = channel();
        Self {
            threads: threads.max(1),
            sps_table: HashMap::new(),
            pps_table: HashMap::new(),
            dpb: Dpb::new(0),
            pending: None,
            open_slices: Vec::new(),
            open: None,
            next_seq: 0,
            commit_next: 0,
            barrier: Arc::new(CommitBarrier {
                committed: Mutex::new(0),
                cv: Condvar::new(),
            }),
            slots: HashMap::new(),
            slot_seqs: HashMap::new(),
            inflight: VecDeque::new(),
            completed: BTreeMap::new(),
            done_rx,
            done_tx,
            first_error: None,
            buffer: Vec::new(),
            gop_id: 0,
            max_depth: 16,
            decoded_frames: AtomicU64::new(0),
        }
    }

    /// Number of worker threads this decoder may use.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Total frames decoded and committed so far.
    pub fn decoded_frames(&self) -> u64 {
        self.decoded_frames.load(Ordering::Relaxed)
    }

    /// Frame rate from the most recently parsed SPS's VUI timing info, as
    /// `(numerator, denominator)`; `None` when no timing info is present.
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        self.sps_table.values().find_map(|s| s.frame_rate())
    }

    /// Frame rate as a single floating-point value.
    pub fn frame_rate_f64(&self) -> Option<f64> {
        let (n, d) = self.frame_rate()?;
        Some(n as f64 / d as f64)
    }

    /// Feed one NAL unit; returns any frames that have become ready, in
    /// display order. Byte-identical to feeding the same NALs to
    /// [`OrderedDecoder`](crate::decoder::OrderedDecoder).
    pub fn decode_nal(&mut self, nal: &NalUnit) -> Result<Vec<Frame>, DecodeError> {
        let mut out = Vec::new();
        match nal.nal_unit_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&nal.rbsp)?;
                self.dpb.set_max_ref_frames(sps.max_num_ref_frames);
                self.sps_table.insert(sps.seq_parameter_set_id, sps);
            }
            NalUnitType::Pps => {
                let pps_id_sps = {
                    let mut peek = crate::bitstream::BitstreamReader::new(&nal.rbsp);
                    let _ = peek.read_ue();
                    peek.read_ue().ok()
                };
                let sps_ref = pps_id_sps.and_then(|id| self.sps_table.get(&id));
                let pps = parse_pps(&nal.rbsp, sps_ref)?;
                self.pps_table.insert(pps.pic_parameter_set_id, pps);
            }
            NalUnitType::Sei => {}
            NalUnitType::SliceIdr | NalUnitType::Slice => {
                let mut peek = crate::bitstream::BitstreamReader::new(&nal.rbsp);
                let first_mb = peek.read_ue().unwrap_or(0);
                let is_new_picture = first_mb == 0;

                if is_new_picture {
                    // Dispatch the previously-open picture, then commit
                    // whatever has finished.
                    self.close_open_picture()?;
                    self.pump(&mut out);
                    while self.inflight.len() >= self.threads {
                        self.wait_one_completion();
                        self.pump(&mut out);
                    }
                }

                // Prepare this slice against the shadow DPB. Only
                // continuation slices can consume `pending`, so only those
                // need the restore-on-error backup (a full picture clone).
                let pending_before = if first_mb > 0 {
                    self.pending.clone()
                } else {
                    None
                };
                match prepare_slice_job(
                    &self.sps_table,
                    &self.pps_table,
                    &mut self.dpb,
                    &mut self.pending,
                    nal,
                ) {
                    Ok((shell, ps)) => {
                        if shell.header.field_pic_flag {
                            // Restore pre-slice state; serial decoder handles these.
                            self.pending = pending_before;
                            return Err(DecodeError::InvalidSyntax(
                                "field pictures are not supported by ThreadedDecoder; use Decoder",
                            ));
                        }
                        // IDR boundary for the reorder buffer. Bump BEFORE
                        // recording the picture's gop: OrderedDecoder pushes
                        // the IDR's own frame after the bump (it belongs to
                        // the new GOP), and our output order must match.
                        if is_new_picture && nal.nal_unit_type == NalUnitType::SliceIdr {
                            self.gop_id += 1;
                        }
                        if !shell.is_continuation {
                            // Assign the picture's pipeline identity here; the
                            // shadow-DPB insert happens at close time (below),
                            // matching the serial decoder's finalize ordering
                            // so this picture is never in its own ref lists.
                            let seq = self.next_seq;
                            self.next_seq += 1;
                            let slot: SharedSlot = Arc::new(OnceLock::new());
                            self.slots.insert(seq, (slot.clone(), self.gop_id));
                            self.open = Some(OpenPicture { seq, slot });
                        }
                        self.pending = Some(ps);
                        self.open_slices.push(ThreadSlice {
                            shell,
                            rbsp: nal.rbsp.to_vec(),
                        });
                    }
                    Err(e) => {
                        self.pending = pending_before.or(self.pending.take());
                        return Err(e);
                    }
                }
            }
            _ => {}
        }
        self.pump(&mut out);
        if let Some(e) = self.first_error.take() {
            return Err(e);
        }
        Ok(out)
    }

    /// Wait for all in-flight pictures, commit everything in order, and
    /// return all remaining frames in display order.
    pub fn flush(&mut self) -> Vec<Frame> {
        let _ = self.close_open_picture();
        let mut out = Vec::new();
        self.pump(&mut out);
        // Join workers in coded order, pumping between joins: a worker may
        // be waiting on the commit barrier, and commits happen in pump().
        while !self.inflight.is_empty() {
            let handle = self.inflight.pop_front().unwrap();
            let _ = handle.join();
            self.pump(&mut out);
        }
        if self.first_error.is_some() {
            // Surface the error but still return any decoded frames.
            self.first_error = None;
        }
        self.buffer.sort_by_key(|(g, f)| (*g, f.pic_order_cnt));
        out.extend(self.buffer.drain(..).map(|(_, f)| f));
        out
    }

    /// Dispatch the open picture (if any) to a worker thread.
    fn close_open_picture(&mut self) -> Result<(), DecodeError> {
        let Some(open) = self.open.take() else {
            return Ok(());
        };
        let Some(ps) = self.pending.take() else {
            return Ok(());
        };
        let slices = std::mem::take(&mut self.open_slices);

        // Speculative reference marking on the shadow DPB, exactly the
        // sequence the serial decoder applies when finalizing this picture
        // (i.e. when the next picture starts). Uses the last slice's header,
        // like the serial path. Until this runs, the picture is not in the
        // shadow DPB, so continuation slices never see it in their lists.
        let last = slices.last().expect("open picture has slices");
        let planned = PlannedPic {
            slot: open.slot.clone(),
            poc: last.shell.current_poc,
            frame_num: last.shell.header.frame_num,
            structure: PictureStructure::Frame,
        };
        apply_reference_marking(
            &mut self.dpb,
            planned,
            last.shell.nal_unit_type,
            last.shell.nal_ref_idc,
            last.shell.header.frame_num,
            false,
            &last.shell.header.mmco_ops,
            last.shell.header.long_term_reference_flag,
        );
        self.slot_seqs
            .insert(Arc::as_ptr(&open.slot) as usize, open.seq);

        // Commit dependency: the newest reference picture of any slice.
        let needed = slices
            .iter()
            .flat_map(|s| {
                s.shell
                    .ref_pic_list
                    .iter()
                    .chain(s.shell.ref_pic_list_l0.iter())
                    .chain(s.shell.ref_pic_list_l1.iter())
            })
            .filter_map(|p| self.slot_seqs.get(&(Arc::as_ptr(&p.slot) as usize)).copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let barrier = Arc::clone(&self.barrier);
        let done_tx = self.done_tx.clone();
        let seq = open.seq;
        let worker = move || {
            // Wait until all reference pictures are committed.
            {
                let mut committed = barrier.committed.lock().unwrap();
                while *committed < needed {
                    committed = barrier.cv.wait(committed).unwrap();
                }
            }
            // Materialize reference lists (slots are guaranteed full now).
            let result = run_picture(&slices, ps);
            let _ = done_tx.send((seq, result));
        };
        let handle = std::thread::spawn(worker);
        self.inflight.push_back(handle);
        Ok(())
    }

    /// Move finished worker results into `completed` and commit as many
    /// pictures in order as possible, emitting display-ready frames.
    fn pump(&mut self, out: &mut Vec<Frame>) {
        while let Ok((seq, res)) = self.done_rx.try_recv() {
            self.completed.insert(seq, res);
        }
        // Drop finished worker handles so the in-flight bound reflects
        // actually-running workers (their results are already queued).
        self.inflight.retain(|h| !h.is_finished());
        while self.commit_next < self.next_seq {
            let seq = self.commit_next;
            let Some(res) = self.completed.remove(&seq) else {
                break;
            };
            let (slot, gop) = self
                .slots
                .remove(&seq)
                .expect("slot for scheduled picture");
            match res {
                Ok(mut ps) => {
                    // Output frame: cropped copy of the coded picture.
                    // DPB picture: the coded planes, moved (same number of
                    // full-frame copies as the serial path, which clones
                    // into the Arc and crops its own copy in place).
                    let coded_w = (ps.mb_width * 16) as usize;
                    let mut frame = ps.frame.clone();
                    crop_to_display(&mut frame, coded_w);
                    let pic = take_decoded_picture(&mut ps);
                    let _ = slot.set(pic);
                    self.decoded_frames.fetch_add(1, Ordering::Relaxed);
                    self.buffer.push((gop, frame));
                    // OrderedDecoder drains a completed GOP as one
                    // POC-sorted batch when the IDR NAL arrives. Commits are
                    // coded-ordered, so the first commit of a new GOP proves
                    // every older GOP is complete — drain them here, sorted.
                    self.drain_stale_gops(gop, out);
                    while self.buffer.len() > self.max_depth {
                        let f = self.pop_lowest();
                        out.push(f);
                    }
                }
                Err(e) => {
                    if self.first_error.is_none() {
                        self.first_error = Some(e);
                    }
                    // Keep the pipeline alive: publish an empty picture so
                    // dependent frames can still decode.
                    let _ = slot.set(Arc::new(DecodedPicture {
                        y: Vec::new(),
                        u: Vec::new(),
                        v: Vec::new(),
                        width: 0,
                        height: 0,
                        frame_num: 0,
                        pic_order_cnt: 0,
                        mv_l0: Vec::new(),
                        ref_idx_l0: Vec::new(),
                        ref_poc_l0: Vec::new(),
                        mv_l1: Vec::new(),
                        ref_idx_l1: Vec::new(),
                        mb_width: 0,
                        is_intra: false,
                        structure: PictureStructure::Frame,
                    }));
                }
            }
            // Advance the commit barrier and wake waiting workers.
            {
                let mut committed = self.barrier.committed.lock().unwrap();
                *committed = seq + 1;
                self.barrier.cv.notify_all();
            }
            self.commit_next = seq + 1;
        }
    }

    /// Block until a worker result arrives (or a short timeout passes).
    ///
    /// The timeout is required: a worker may have sent its result — already
    /// drained by an earlier `pump` — while its handle still reads as
    /// unfinished, leaving nothing for `recv` to wait for. On timeout the
    /// caller's `pump` re-checks handle state and commits.
    fn wait_one_completion(&mut self) {
        match self.done_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok((seq, res)) => {
                self.completed.insert(seq, res);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // A worker panicked; join to surface it.
                while let Some(handle) = self.inflight.pop_front() {
                    let _ = handle.join();
                }
            }
        }
    }

    /// Drain buffered frames from GOPs older than `pushed_gop`, as one
    /// `(gop, poc)`-sorted batch — the timing-independent equivalent of
    /// OrderedDecoder's drain at the IDR NAL.
    fn drain_stale_gops(&mut self, pushed_gop: u32, out: &mut Vec<Frame>) {
        let has_stale = self.buffer.iter().any(|(g, _)| *g < pushed_gop);
        if !has_stale {
            return;
        }
        let mut completed: Vec<(u32, Frame)> = Vec::new();
        let mut remaining: Vec<(u32, Frame)> = Vec::with_capacity(self.buffer.len());
        for entry in self.buffer.drain(..) {
            if entry.0 < pushed_gop {
                completed.push(entry);
            } else {
                remaining.push(entry);
            }
        }
        completed.sort_by_key(|(g, f)| (*g, f.pic_order_cnt));
        out.extend(completed.into_iter().map(|(_, f)| f));
        self.buffer = remaining;
    }

    fn pop_lowest(&mut self) -> Frame {
        let idx = self
            .buffer
            .iter()
            .enumerate()
            .min_by_key(|(_, (g, f))| (*g, f.pic_order_cnt))
            .map(|(i, _)| i)
            .unwrap();
        self.buffer.remove(idx).1
    }
}

/// Worker-side: run all slices of one picture, then deblock.
fn run_picture(slices: &[ThreadSlice], mut ps: PictureState) -> Result<PictureState, DecodeError> {
    for (i, slice) in slices.iter().enumerate() {
        // Materialize real reference pictures for this slice.
        let job: SliceJob = materialize(&slice.shell);
        if i == 0 {
            match run_slice_job(&job, ps, &slice.rbsp) {
                Ok(done) => ps = done,
                Err(e) => return Err(e),
            }
        } else {
            // Continuation: keep a backup so an end-of-slice error doesn't
            // lose already-decoded MBs (mirrors the serial decoder).
            let backup = ps.clone();
            match run_slice_job(&job, ps, &slice.rbsp) {
                Ok(done) => ps = done,
                Err(_) => {
                    ps = backup;
                    break;
                }
            }
        }
    }
    deblock_picture(&mut ps);
    Ok(ps)
}

/// Convert a prepared slice with planned references into one with real
/// reference pictures. Only called after the commit barrier guarantees the
/// slots are filled.
fn materialize(shell: &SliceJobShell<PlannedPic>) -> SliceJob {
    let conv = |l: &[PlannedPic]| -> Vec<Arc<DecodedPicture>> {
        l.iter()
            .map(|p| {
                p.slot
                    .get()
                    .expect("reference picture slot not filled")
                    .clone()
            })
            .collect()
    };
    SliceJobShell {
        nal_unit_type: shell.nal_unit_type,
        nal_ref_idc: shell.nal_ref_idc,
        sps: shell.sps.clone(),
        pps: shell.pps.clone(),
        header: shell.header.clone(),
        ref_pic_list: conv(&shell.ref_pic_list),
        ref_pic_list_l0: conv(&shell.ref_pic_list_l0),
        ref_pic_list_l1: conv(&shell.ref_pic_list_l1),
        implicit_weights: shell.implicit_weights.clone(),
        use_weight: shell.use_weight,
        current_poc: shell.current_poc,
        slice_qp: shell.slice_qp,
        is_continuation: shell.is_continuation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::parse_annex_b;

    /// Decode a stream serially (OrderedDecoder) and threaded, compare the
    /// full display-order frame streams byte-for-byte.
    fn compare_threaded(path: &str, threads: usize) {
        let data = std::fs::read(path).unwrap();
        let nals = parse_annex_b(&data);

        let mut serial = crate::decoder::OrderedDecoder::new();
        let mut expect: Vec<Vec<u8>> = Vec::new();
        for nal in &nals {
            for f in serial.decode_nal(nal).unwrap() {
                let mut bytes = f.y.clone();
                bytes.extend_from_slice(&f.u);
                bytes.extend_from_slice(&f.v);
                expect.push(bytes);
            }
        }
        for f in serial.flush() {
            let mut bytes = f.y.clone();
            bytes.extend_from_slice(&f.u);
            bytes.extend_from_slice(&f.v);
            expect.push(bytes);
        }

        let mut threaded = ThreadedDecoder::new(threads);
        let mut got: Vec<Vec<u8>> = Vec::new();
        for nal in &nals {
            let emitted = match threaded.decode_nal(nal) {
                Ok(fs) => fs,
                Err(crate::error::DecodeError::InvalidSyntax(
                    "field pictures are not supported by ThreadedDecoder; use Decoder",
                )) => {
                    // Interlaced-field streams are documented as unsupported.
                    return;
                }
                Err(e) => panic!("{path}: unexpected decode error: {e:?}"),
            };
            for f in emitted {
                let mut bytes = f.y.clone();
                bytes.extend_from_slice(&f.u);
                bytes.extend_from_slice(&f.v);
                got.push(bytes);
            }
        }
        for f in threaded.flush() {
            let mut bytes = f.y.clone();
            bytes.extend_from_slice(&f.u);
            bytes.extend_from_slice(&f.v);
            got.push(bytes);
        }

        assert_eq!(
            expect.len(),
            got.len(),
            "{path}: frame count mismatch (serial {}, threaded {})",
            expect.len(),
            got.len()
        );
        for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
            assert_eq!(e, g, "{path}: frame {i} differs between serial and threaded");
        }
    }

    fn multiframe_streams() -> Vec<String> {
        let dir = format!("{}/testdata", env!("CARGO_MANIFEST_DIR"));
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("h264") {
                    let name = p.file_name().unwrap().to_string_lossy().to_string();
                    // Only multi-frame streams exercise the pipeline.
                    if name.contains("frame")
                        || name.contains("b_")
                        || name.contains("p_")
                        || name.contains("mbaff")
                        || name.contains("multi")
                        || name.contains("bench")
                        || name.contains("realworld")
                        || name.contains("1080p")
                        || name.contains("720p")
                    {
                        out.push(p.to_string_lossy().to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn threaded_matches_serial_2_threads() {
        for stream in multiframe_streams() {
            compare_threaded(&stream, 2);
        }
    }

    #[test]
    fn threaded_matches_serial_4_threads() {
        for stream in multiframe_streams() {
            compare_threaded(&stream, 4);
        }
    }
}
