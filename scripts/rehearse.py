"""Run ONE model in a fresh process to release accelerator memory on exit."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import BASELINES, FOUNDATIONS, OPTIONAL, load_series, save_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, choices=(*BASELINES, *FOUNDATIONS, *OPTIONAL))
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
print(save_run(
    args.model, load_series(), args.output, args.device
).to_string(index=False))
print(f"Saved real results to {args.output}.")