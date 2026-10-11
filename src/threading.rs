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
    Frame, PictureState, SliceJob, SliceJobShell,
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

/// One picture-decode task for the worker pool.
struct PoolTask {
    seq: u64,
    slices: Vec<ThreadSlice>,
    ps: PictureState,
    shared: Arc<DecodedPicture>,
}

/// Shared worker-pool state: a task queue with shutdown flag. Workers park
/// on the condvar when the queue is empty and exit once shutdown is set
/// and the queue is drained.
struct Pool {
    queue: Mutex<VecDeque<PoolTask>>,
    cv: Condvar,
    shutdown: Mutex<bool>,
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
    /// GOP id per scheduled seq, for the reorder buffer at commit time.
    slots: HashMap<u64, u32>,
    /// Slot identity -> seq, to compute a picture's commit dependencies.
    slot_seqs: HashMap<usize, u64>,
    /// Worker pool (created lazily on the first dispatched picture).
    pool: Option<Arc<Pool>>,
    worker_handles: Vec<JoinHandle<()>>,
    /// Pictures scheduled but whose results have not been received yet
    /// (running or queued in the pool). Bounds the pipeline like the old
    /// in-flight handle count did.
    unfinished: usize,
    completed: BTreeMap<u64, Result<Frame, DecodeError>>,
    done_rx: Receiver<(u64, Result<Frame, DecodeError>)>,
    done_tx: Sender<(u64, Result<Frame, DecodeError>)>,
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
            slots: HashMap::new(),
            slot_seqs: HashMap::new(),
            pool: None,
            worker_handles: Vec::new(),
            unfinished: 0,
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
                    while self.unfinished >= self.threads {
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
                            self.slots.insert(seq, self.gop_id);
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
        // Receive every outstanding result (workers may still be running;
        // pumping between receives commits in coded order). Workers
        // themselves stay pooled for reuse until the decoder is dropped.
        while self.unfinished > 0 {
            self.wait_one_completion();
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
        // sequence the serial decoder applies when finalizing this picture.
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

        // Allocate the shared picture and publish it in the slot now:
        // dependent workers materialize it immediately and wait per row
        // during motion compensation instead of waiting for the whole
        // frame (row-level reference synchronization).
        let mb_w = ps.mb_width as usize;
        let mb_h = ps.mb_height as usize;
        let stride = mb_w * 16;
        let blocks = mb_w * mb_h * 16;
        let pic = Arc::new(DecodedPicture {
            y: vec![0u8; stride * mb_h * 16],
            u: vec![0u8; (stride / 2) * (mb_h * 8)],
            v: vec![0u8; (stride / 2) * (mb_h * 8)],
            width: stride as u32,
            height: (mb_h * 16) as u32,
            frame_num: ps.frame_num,
            pic_order_cnt: ps.poc,
            mv_l0: vec![[0i16; 2]; blocks],
            ref_idx_l0: vec![-1i8; blocks],
            ref_poc_l0: vec![-1i32; blocks],
            mv_l1: vec![[0i16; 2]; blocks],
            ref_idx_l1: vec![-1i8; blocks],
            mb_width: ps.mb_width,
            is_intra: ps.is_intra_slice,
            structure: PictureStructure::Frame,
            row_progress: std::sync::atomic::AtomicUsize::new(0),
        });
        let _ = open.slot.set(Arc::clone(&pic));

        // Lazily create the worker pool on first use, then queue the task.
        let pool = self.pool.get_or_insert_with(|| {
            let pool = Arc::new(Pool {
                queue: Mutex::new(VecDeque::new()),
                cv: Condvar::new(),
                shutdown: Mutex::new(false),
            });
            let mut handles = Vec::with_capacity(self.threads);
            for _ in 0..self.threads {
                let pool = Arc::clone(&pool);
                let done_tx = self.done_tx.clone();
                handles.push(std::thread::spawn(move || pool_worker(pool, done_tx)));
            }
            self.worker_handles = handles;
            pool
        });
        {
            let mut q = pool.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.push_back(PoolTask {
                seq: open.seq,
                slices,
                ps,
                shared: Arc::clone(&pic),
            });
        }
        self.unfinished += 1;
        pool.cv.notify_one();
        Ok(())
    }

    /// Move finished worker results into `completed` and commit as many
    /// pictures in order as possible, emitting display-ready frames.
    fn pump(&mut self, out: &mut Vec<Frame>) {
        while let Ok((seq, res)) = self.done_rx.try_recv() {
            self.unfinished = self.unfinished.saturating_sub(1);
            self.completed.insert(seq, res);
        }
        while self.commit_next < self.next_seq {
            let seq = self.commit_next;
            let Some(res) = self.completed.remove(&seq) else {
                break;
            };
            let gop = self.slots.remove(&seq).expect("gop for scheduled picture");
            match res {
                Ok(frame) => {
                    self.decoded_frames.fetch_add(1, Ordering::Relaxed);
                    self.buffer.push((gop, frame));
                    // Match OrderedDecoder's per-NAL order exactly: drain
                    // completed GOPs first, then enforce the depth bound.
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
                }
            }
            self.commit_next = seq + 1;
        }
    }

    /// Block until a worker result arrives (or a short timeout passes).
    fn wait_one_completion(&mut self) {
        match self.done_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok((seq, res)) => {
                self.unfinished = self.unfinished.saturating_sub(1);
                self.completed.insert(seq, res);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Cannot happen while `self.done_tx` exists; a panicking
                // worker is contained by the pool's catch_unwind.
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

/// Worker-side: run all slices of one picture, publishing rows into the
/// shared picture as the decode loop completes them (deblock lag included),
/// then deblock the tail and return the cropped output frame.
///
/// On error the shared picture's progress is set to complete so dependent
/// workers blocked on its rows read the (zero) pixels instead of hanging —
/// the same garbage-in behavior as the serial decoder's error path.
fn run_picture(
    slices: &[ThreadSlice],
    mut ps: PictureState,
    shared: &Arc<DecodedPicture>,
) -> Result<Frame, DecodeError> {
    let mb_w = ps.mb_width as usize;
    let result = run_picture_inner(slices, &mut ps, shared);
    if let Err(e) = result {
        shared
            .row_progress
            .store(usize::MAX, std::sync::atomic::Ordering::Release);
        return Err(e);
    }
    let mut frame = ps.frame;
    crop_to_display(&mut frame, mb_w * 16);
    Ok(frame)
}

fn run_picture_inner(
    slices: &[ThreadSlice],
    ps: &mut PictureState,
    shared: &Arc<DecodedPicture>,
) -> Result<(), DecodeError> {
    let mut published = 0usize;

    {
        let shared = Arc::clone(shared);
        let mut on_row = |r: usize,
                          frame: &Frame,
                          mv_l0: &[[i16; 2]],
                          ri_l0: &[i8],
                          rp_l0: &[i32],
                          mv_l1: &[[i16; 2]],
                          ri_l1: &[i8]| {
            if r != published {
                return; // rows arrive in order; ignore re-announcements
            }
            publish_row(&shared, r, frame, mv_l0, ri_l0, rp_l0, mv_l1, ri_l1);
            published = r + 1;
        };

        for (i, slice) in slices.iter().enumerate() {
            let job: SliceJob = materialize(&slice.shell);
            let taken = std::mem::replace(ps, PictureState::empty());
            if i == 0 {
                match run_slice_job(&job, taken, &slice.rbsp, &mut on_row) {
                    Ok(done) => *ps = done,
                    Err(e) => return Err(e),
                }
            } else {
                // `ps` is the empty replacement; retain the valid picture
                // taken above so a malformed continuation cannot erase it.
                let backup = taken.clone();
                match run_slice_job(&job, taken, &slice.rbsp, &mut on_row) {
                    Ok(done) => *ps = done,
                    Err(_) => {
                        *ps = backup;
                        break;
                    }
                }
            }
        }
    }

    deblock_picture(ps);
    // Publish the tail rows (the deblock lag leaves up to two unpublished)
    // and mark the picture complete.
    let mb_h = ps.mb_height as usize;
    for r in published..mb_h {
        publish_row(
            shared,
            r,
            &ps.frame,
            &ps.mv_store_l0,
            &ps.ref_idx_store_l0,
            &ps.ref_poc_store_l0,
            &ps.mv_store_l1,
            &ps.ref_idx_store_l1,
        );
    }
    shared
        .row_progress
        .store(usize::MAX, std::sync::atomic::Ordering::Release);
    Ok(())
}

/// Copy one MB row's pixels and MV arrays into the shared picture, then
/// publish it with a Release store (readers acquire-load `row_progress`
/// before touching the bytes).
///
/// # Safety contract
/// Single writer (the owning worker) plus readers that have observed
/// `row_progress > r*16` via acquire. Rows below `r` are never written
/// again, so overlapping readers of earlier rows are sound.
#[allow(clippy::too_many_arguments)]
fn publish_row(
    shared: &Arc<DecodedPicture>,
    r: usize,
    frame: &Frame,
    mv_l0: &[[i16; 2]],
    ri_l0: &[i8],
    rp_l0: &[i32],
    mv_l1: &[[i16; 2]],
    ri_l1: &[i8],
) {
    let mb_w = shared.mb_width as usize;
    let stride = mb_w * 16;
    let c_stride = stride / 2;

    unsafe fn copy_bytes(dst: *const u8, src: *const u8, off: usize, len: usize) {
        std::ptr::copy_nonoverlapping(src, dst.add(off) as *mut u8, len);
    }
    // SAFETY: the shared planes are sized for exactly these ranges; see the
    // safety contract above.
    unsafe {
        for dy in 0..16 {
            let y = r * 16 + dy;
            copy_bytes(
                shared.y.as_ptr(),
                frame.y.as_ptr().add(y * stride),
                y * stride,
                stride,
            );
        }
        for dy in 0..8 {
            let cy = r * 8 + dy;
            copy_bytes(
                shared.u.as_ptr(),
                frame.u.as_ptr().add(cy * c_stride),
                cy * c_stride,
                c_stride,
            );
            copy_bytes(
                shared.v.as_ptr(),
                frame.v.as_ptr().add(cy * c_stride),
                cy * c_stride,
                c_stride,
            );
        }
        let base = r * mb_w * 16;
        let len = mb_w * 16;
        copy_bytes(
            shared.mv_l0.as_ptr() as *const u8,
            (mv_l0.as_ptr() as *const u8).add(base * 4),
            base * 4,
            len * 4,
        );
        copy_bytes(
            shared.ref_idx_l0.as_ptr() as *const u8,
            (ri_l0.as_ptr() as *const u8).add(base),
            base,
            len,
        );
        copy_bytes(
            shared.ref_poc_l0.as_ptr() as *const u8,
            (rp_l0.as_ptr() as *const u8).add(base * 4),
            base * 4,
            len * 4,
        );
        copy_bytes(
            shared.mv_l1.as_ptr() as *const u8,
            (mv_l1.as_ptr() as *const u8).add(base * 4),
            base * 4,
            len * 4,
        );
        copy_bytes(
            shared.ref_idx_l1.as_ptr() as *const u8,
            (ri_l1.as_ptr() as *const u8).add(base),
            base,
            len,
        );
    }
    shared
        .row_progress
        .store((r + 1) * 16, std::sync::atomic::Ordering::Release);
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

/// Pool worker loop: park on the queue condvar, run tasks, send results.
/// A panicking task is caught and reported as an error (with the shared
/// picture marked complete so row-waiters do not spin forever).
fn pool_worker(pool: Arc<Pool>, done_tx: Sender<(u64, Result<Frame, DecodeError>)>) {
    loop {
        let task = {
            let mut q = pool.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(task) = q.pop_front() {
                    break task;
                }
                if *pool.shutdown.lock().unwrap_or_else(|e| e.into_inner()) {
                    return;
                }
                q = pool.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        let PoolTask { seq, slices, ps, shared: pic } = task;
        let guard = Arc::clone(&pic);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_picture(&slices, ps, &pic)
        }))
        .unwrap_or_else(|_| {
            guard
                .row_progress
                .store(usize::MAX, std::sync::atomic::Ordering::Release);
            Err(DecodeError::InvalidSyntax("decode worker panicked"))
        });
        let _ = done_tx.send((seq, result));
    }
}

impl Drop for ThreadedDecoder {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            {
                let mut q = pool.queue.lock().unwrap_or_else(|e| e.into_inner());
                q.clear();
            }
            {
                let mut flag = pool.shutdown.lock().unwrap_or_else(|e| e.into_inner());
                *flag = true;
            }
            pool.cv.notify_all();
        }
        for h in self.worker_handles.drain(..) {
            let _ = h.join();
        }
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
    fn threaded_mc_progressive_and_mbaff_parity() {
        for name in [
            "p_multi_frame",
            "mbaff_field_p_test",
            "mbaff_field_cabac_test",
        ] {
            compare_threaded(
                &format!("{}/testdata/{name}.h264", env!("CARGO_MANIFEST_DIR")),
                2,
            );
        }
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

    // Small self-contained Baseline streams keep this regression independent
    // of the external corpus and avoid reference-picture dependencies.
    fn sequence_test_nal(kind: NalUnitType, bits: &str) -> NalUnit<'static> {
        let mut bits = bits.to_owned();
        bits.push('1'); // rbsp_stop_one_bit
        while bits.len() % 8 != 0 {
            bits.push('0');
        }
        let rbsp = bits
            .as_bytes()
            .chunks(8)
            .map(|byte| byte.iter().fold(0u8, |v, &bit| (v << 1) | (bit - b'0')))
            .collect::<Vec<_>>();
        NalUnit {
            nal_ref_idc: 0,
            nal_unit_type: kind,
            rbsp: rbsp.into(),
        }
    }

    fn sequence_test_ue(value: u32) -> String {
        let suffix = format!("{:b}", value + 1);
        format!("{}{}", "0".repeat(suffix.len() - 1), suffix)
    }

    fn sequence_test_sps(mb_width: u32, mb_height: u32) -> NalUnit<'static> {
        // Baseline, level 1, SPS 0, four-bit frame_num, POC type 2,
        // one reference, progressive frame, no cropping/VUI.
        sequence_test_nal(
            NalUnitType::Sps,
            &format!(
                "010000100000000000001010110110100{}{}1100",
                sequence_test_ue(mb_width - 1),
                sequence_test_ue(mb_height - 1)
            ),
        )
    }

    fn sequence_test_slice(first_mb: u32, count: usize, sample: u8) -> NalUnit<'static> {
        // Non-reference I slice, PPS 0, frame_num 0, QP delta 0,
        // deblocking disabled. Each macroblock is I_PCM.
        let mut bits = format!("{}011100001010", sequence_test_ue(first_mb));
        for _ in 0..count {
            bits.push_str(&sequence_test_ue(25));
            while bits.len() % 8 != 0 {
                bits.push('0');
            }
            bits.push_str(&format!("{sample:08b}").repeat(384));
        }
        sequence_test_nal(NalUnitType::Slice, &bits)
    }

    #[test]
    fn malformed_continuation_retains_pixels_and_completes_progress() {
        let sps = sequence_test_sps(2, 2);
        let pps = sequence_test_nal(NalUnitType::Pps, "1100111000111100");
        let first = sequence_test_slice(0, 2, 61);
        let mut malformed = sequence_test_slice(2, 1, 93);
        malformed.rbsp = malformed.rbsp[..8].to_vec().into();
        let mut decoder = ThreadedDecoder::new(1);
        for nal in [&sps, &pps, &first, &malformed] {
            decoder.decode_nal(nal).unwrap();
        }
        let slot = decoder.open.as_ref().unwrap().slot.clone();
        decoder.close_open_picture().unwrap();
        let picture = slot.get().unwrap().clone();
        let frames = decoder.flush();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].width, frames[0].height), (32, 32));
        assert!(frames[0].y[..32 * 16].iter().all(|&v| v == 61));
        assert!(frames[0].u[..16 * 8].iter().all(|&v| v == 61));
        assert!(frames[0].v[..16 * 8].iter().all(|&v| v == 61));
        assert_eq!(
            picture
                .row_progress
                .load(std::sync::atomic::Ordering::Acquire),
            usize::MAX
        );
        assert!(decoder.slots.is_empty());
        assert_eq!(decoder.unfinished, 0);
    }

    #[test]
    fn incompatible_continuation_preserves_sequence_progress() {
        let sps = sequence_test_sps(2, 1);
        let changed_sps = sequence_test_sps(3, 1);
        let reshaped_sps = sequence_test_sps(1, 2);
        assert_eq!(parse_sps(&sps.rbsp).unwrap().width(), 32);
        assert_eq!(parse_sps(&changed_sps.rbsp).unwrap().width(), 48);
        let pps = sequence_test_nal(NalUnitType::Pps, "1100111000111100");
        let first = sequence_test_slice(0, 1, 61);
        let continuation = sequence_test_slice(1, 1, 93);
        let full = sequence_test_slice(0, 2, 117);
        let mut decoder = ThreadedDecoder::new(1);
        let mut serial = crate::decoder::OrderedDecoder::new();
        for nal in [&sps, &pps, &first] {
            serial.decode_nal(nal).unwrap();
        }
        decoder.decode_nal(&sps).unwrap();
        decoder.decode_nal(&pps).unwrap();
        decoder.decode_nal(&first).unwrap();
        for incompatible in [&changed_sps, &reshaped_sps] {
            decoder.decode_nal(incompatible).unwrap();
            serial.decode_nal(incompatible).unwrap();
            let result = decoder.decode_nal(&continuation);
            assert!(
                matches!(
                    result,
                    Err(DecodeError::InvalidSyntax(
                        "continuation slice is incompatible with the open picture"
                    ))
                ),
                "result={result:?}, next_seq={}, open_seq={:?}",
                decoder.next_seq,
                decoder.open.as_ref().map(|p| p.seq)
            );
            // Serial recovery absorbs continuation errors while preserving pending.
            assert!(serial.decode_nal(&continuation).unwrap().is_empty());
        }
        assert_eq!(decoder.next_seq, 1);
        assert_eq!(decoder.open.as_ref().unwrap().seq, 0);
        assert_eq!(decoder.open_slices.len(), 1);

        // Restore the SPS and finish the original picture, then feed enough
        // pictures to expose a missing terminal result as growing storage.
        decoder.decode_nal(&sps).unwrap();
        decoder.decode_nal(&continuation).unwrap();
        serial.decode_nal(&sps).unwrap();
        serial.decode_nal(&continuation).unwrap();
        let mut expected = Vec::new();
        let mut frames = Vec::new();
        for _ in 0..64 {
            frames.extend(decoder.decode_nal(&full).unwrap());
            expected.extend(serial.decode_nal(&full).unwrap());
            assert!(decoder.completed.len() <= 1);
            assert!(decoder.slots.len() <= 2);
        }
        frames.extend(decoder.flush());
        expected.extend(serial.flush());
        assert_eq!(frames.len(), expected.len());
        for (actual, expected) in frames.iter().zip(&expected) {
            assert_eq!(actual.y, expected.y);
            assert_eq!(actual.u, expected.u);
            assert_eq!(actual.v, expected.v);
            assert_eq!(actual.pic_order_cnt, expected.pic_order_cnt);
        }
        assert_eq!(frames.len(), 65);
        assert_eq!(frames[0].y[0], 61);
        assert_eq!(frames[0].y[16], 93);
        assert!(frames[1..]
            .iter()
            .all(|frame| frame.y.iter().all(|&v| v == 117)));
        assert_eq!(decoder.commit_next, decoder.next_seq);
        assert_eq!(decoder.decoded_frames(), 65);
        assert!(decoder.completed.is_empty());
        assert!(decoder.slots.is_empty());
        assert!(decoder.flush().is_empty());
    }
}
