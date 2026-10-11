"""Generate the checked 64x64 CABAC MBAFF regression fixture.

Requires FFmpeg with libx264. The existing 64x32 generator contains the
analytic source and encoding recipe; this wrapper selects the 64x64 case.
Executables are used only to encode and independently decode the stream.
"""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path(__file__).with_name("generate_mbaff_skipped_top_field_bottom.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "target/spec-audit",
        help="output directory (defaults to ignored target/spec-audit)",
    )
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("mbaff_generator", GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.HEIGHT = 64
    generator.STEM = "mbaff_cabac_second_pair_row_64x64_test"
    generator.OUT = args.out.resolve()
    generator.OUT.mkdir(parents=True, exist_ok=True)
    generator.main()


if __name__ == "__main__":
    main()
