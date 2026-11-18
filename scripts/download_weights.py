"""Download the pretrained model weights into the Hugging Face cache (about 2.3 GB)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import CHECKPOINTS, FOUNDATIONS  # import first: it disables HF telemetry
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("models", nargs="*", choices=FOUNDATIONS, default=list(FOUNDATIONS))
args = parser.parse_args()
for name in args.models:
    print(f"Downloading {name} ({CHECKPOINTS[name]}) ...")
    snapshot_download(CHECKPOINTS[name])
print("WEIGHTS READY: later runs of these models need no download.")
