---
name: prepare-tsfm-workshop
description: Prepare the environment for the Can AI Predict the Future time-series foundation-model workshop. Use when asked "prepare me for foundation models for time series workshop" or to install/check this workshop's sktime, Colab, or local dependencies.
---

# Prepare the time-series foundation-model workshop

Read the workshop's `README.md`, `PARTICIPANT_PREP.md`, `SETUP.md`, and `pyproject.toml` first.
Locate the actual workshop directory; do not invent repository URLs or assume
the working directory contains it. If materials are absent, request their path.

Ask one focused question if the execution track is unspecified: Colab,
local CPU baselines (recommended for laptops), or local GPU foundation models.
For Colab, provide/run the notebook bootstrap and installation cells in the
user's Colab environment; do not imply a local shell controls their Colab VM.

Use an isolated project environment. Never install globally, delete an
existing environment, alter system CUDA, start a publicly reachable server,
or modify someone else's files. Check existing work before changing anything.
Use the supplied lockfile and Python 3.12 where practical.

For local core: `uv sync --locked --python 3.12`, then
`uv run python scripts/check_environment.py`.
For local foundation: follow the CPU-wheel or GPU path in `SETUP.md`, then
run `scripts/check_environment.py --foundation` with that environment.
Do not install Kumo or Prophet unless explicitly requested.

An import check is not an inference check. Before downloading large weights,
explain disk/network needs and ask for consent. For t0, separately request
the user's explicit acceptance after linking the actual license; never
accept model terms on their behalf or bypass the adapter's license flag.
Do not request secrets in chat. Use private authentication only if the
current upstream checkpoint actually requires it.

With consent, run the one-model rehearsal commands in separate processes.
Use new output directories rather than overwrite artifacts. Preserve real
predictions, scores, and metadata, and validate them with `workshop.load_run`.
On failure, report the exact model/stage and use the documented baseline or
saved-results path. Do not fake forecasts or declare the model ready merely
because its import works.

Finish with a concise state: core ready/not ready, model imports ready/not
checked, real inference verified/not checked per selected model, artifacts
location, and the exact notebook-launch command. Open Jupyter only if asked,
binding locally; do not leave unnecessary processes running.