"""Rebuild the small, attributed dataset from a downloaded FiveThirtyEight CSV."""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
args = parser.parse_args()
with args.source.open() as handle:
    rows = list(csv.DictReader(handle))
selected = rows[-420:]
dates = [
    date(int(row["year"]), int(row["month"]), int(row["date_of_month"]))
    for row in selected
]
assert len(dates) == 420
assert all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:]))
destination = Path(__file__).resolve().parents[1] / "data/daily_births.csv"
destination.parent.mkdir(exist_ok=True)
with destination.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["date", "births"])
    writer.writerows((day.isoformat(), row["births"]) for day, row in zip(dates, selected))
print(f"Saved {len(selected)} days, {dates[0]} to {dates[-1]}, to {destination}.")