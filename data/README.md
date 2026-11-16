# One dataset: daily U.S. births

`daily_births.csv` contains the final 420 observations of FiveThirtyEight's
`US_births_2000-2014_SSA.csv`: aggregate daily U.S. birth counts originally
provided by the Social Security Administration. This is an educational
forecasting example, not a clinical or staffing recommendation.

**Attribution:** FiveThirtyEight, *U.S. Births*, data supplied by the U.S.
Social Security Administration. Distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: selected the final 420 days, converted the date fields to ISO dates,
and retained only date and birth count. No imputation, resampling, smoothing,
or normalization was performed.

Primary description:
https://github.com/fivethirtyeight/data/tree/master/births

Pinned source:
https://raw.githubusercontent.com/fivethirtyeight/data/4c1ff5e3aef1816ae04af63218015066e186c147/births/US_births_2000-2014_SSA.csv

Repository license:
https://github.com/fivethirtyeight/data/blob/4c1ff5e3aef1816ae04af63218015066e186c147/LICENSE

To reproduce the CSV, download that pinned source to a temporary file, then run:

```bash
python scripts/prepare_data.py path/to/US_births_2000-2014_SSA.csv
```

## Why this choice?

It is small, complete, easy to plot, and has meaningful weekday/weekend patterns
and calendar effects. Daily data retains a visible weekly cycle; weekly
aggregation would erase the very pattern our seasonal-naive model exploits.
The file is bundled so the session does not depend on a data API.

`sktime` works directly with a pandas Series with a regular DatetimeIndex; a
built-in dataset loader is not required. We considered its Monash
`load_forecastingdata` options, including NN5 daily and weekly data. The
missing-free NN5 archive describes imputation using the whole series, which
complicates a leakage-safe lesson. Many tiny built-in examples, such as
`load_airline`, are monthly rather than daily.

## Protocol fixed before seeing model results

Keep the last 42 days out of exploratory analysis. The forecast demonstration
uses the preceding 14 days, with 168 earlier days as context. The benchmark
uses the final 210 days: a 168-day rolling context and three successive,
non-overlapping 14-day test windows, moving the origin by 14 days each time.
In later folds, earlier test outcomes legitimately become observed history.

All models get the same observed target values, horizon, and cutoffs. They may
internally pad/patch/scale differently. No future covariates are provided.
The last 42 days are evaluation data, not a tuning set. If you tune after
looking at them, you need a new untouched test period for a final claim.

This historical public dataset might have appeared in foundation-model
pretraining. We cannot certify that it did not. "Zero-shot" here means no
workshop-specific weight training, not guaranteed unseen pretraining data.
Three folds on one aggregate series are a teaching experiment, not a general
model ranking or an independent replication of vendor benchmarks.
