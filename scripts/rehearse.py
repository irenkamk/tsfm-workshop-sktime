"""Run ONE model in a fresh process to release accelerator memory on exit."""
# TODO: I'd rather move this script in the same directory as workshop.py and call it main.py cause it's entrypoint
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import BASELINES, FOUNDATIONS, load_series, save_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, choices=(*BASELINES, *FOUNDATIONS, "prophet"))
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
# TODO: is this really necessary? It's bit weird to require license acceptance from the command line
parser.add_argument("--accept-t0-license", action="store_true")
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
print(save_run(
    args.model, load_series(), args.output, args.device, args.accept_t0_license
).to_string(index=False))
print(f"Saved real results to {args.output}.")