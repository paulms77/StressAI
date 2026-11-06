# StressAI — Real-Time Stress Detection from Wearable Biosignals

Binary stress/rest classification from two wrist-worn signals — electrodermal
activity (EDA) and blood volume pulse (BVP) — served as a live API with a
streaming dashboard.

The point of the project is not the classifier. It is the path from **raw
wearable CSV → cleaned, aligned, labelled signal → database → trained model →
running service**, with the signal-processing stage carrying most of the weight.

---

## Table of contents

- [Pipeline](#pipeline)
- [Signal processing](#signal-processing)
- [Models](#models)
- [Data](#data)
- [Setup](#setup)
- [Usage](#usage)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

---

## Pipeline

```
Empatica E4 CSVs            stress_data.py
EDA  @  4 Hz  ─┐        ┌─ interpolate missing values
BVP  @ 64 Hz  ─┼───────►│─ upsample EDA 4 → 64 Hz, pad to BVP length
ACC/HR/IBI/TEMP┘        │─ low-pass filter (Chebyshev II / Butterworth)
                        │─ drop constant (sensor-dropout) segments
                        └─ label from the Rest/Task protocol
                                    │
                                    ▼
                           stress_db.py  →  MySQL
                      (subjects, eda_data, bvp_data, label_data)
                                    │
                                    ▼
                             train.py  →  fine_tuned_model.pt
                       sliding window 320 samples (5 s @ 64 Hz)
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                 app/app.py (FastAPI)     dashboard.py (Dash)
                 POST /upload_and_predict  live replay + plots
```

## Signal processing

The two sensors do not agree on anything, which is most of the work.

**Different sampling rates.** EDA arrives at 4 Hz, BVP at 64 Hz. EDA is upsampled
to 64 Hz and padded to the BVP length so the two can be stacked into one
`(timesteps, 2)` tensor. Upsampling the slow signal is preferred to decimating
the fast one — BVP's waveform shape is the informative part and downsampling
destroys it.

**Missing values and dropouts.** Gaps are interpolated per channel. Separately,
long runs of a constant value mark sensor dropout rather than a real flat
signal, and those segments are removed (`remove_constant_segments`, threshold
1e-10) instead of being fed to the model as genuine readings.

**Filtering.** Low-pass filtering (Chebyshev II / Butterworth, zero-phase via
`filtfilt`) removes high-frequency noise. Zero-phase matters here: a causal
filter shifts the signal in time, and a time-shifted signal against fixed
protocol labels means every window is slightly mislabelled.

**Labelling.** The recording protocol alternates rest and task blocks, with
durations in minutes:

| Block | Rest 1 | Task 1 | Rest 2 | Task 2 | Rest 3 | Task 3 | Rest 4 | Task 4 | Rest 5 | Task 5 | Rest 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Minutes | 3 | 10 | 2 | 5 | 2 | 3 | 2 | — | 2 | 1 | 2 |

Samples are labelled by which block they fall in — `Rest → 0`, `Task → 1` —
converting minutes to sample counts at each channel's own rate.

## Models

Both live in [`model.py`](model.py) and share the same input shape,
`(batch, 320, 2)`:

**`StressRNN`** — single-layer LSTM (hidden 64), last hidden state into a linear
head. The straightforward baseline for a time series.

**`StressTransformer`** — linear embedding to 128 dims, 2 `TransformerEncoder`
layers with 4 attention heads, mean-pooled over time, linear head. This is what
is actually served.

Training ([`train.py`](train.py)) runs **per subject, sequentially**, each
subject's model initialised from the previous subject's weights — a simple
sequential fine-tune across 29 subjects rather than one pooled fit. 10 epochs
per subject, Adam at 1e-3, cross-entropy.

## Data

Recordings come from an **Empatica E4** wristband — ACC, BVP, EDA, HR, IBI and
TEMP for 29 participants.

> **The biosignal recordings are not included in this repository.** They are
> physiological measurements from human participants, so they are not
> redistributed here. The code expects them under `data/` — see
> [`data/README.md`](data/README.md) for the directory layout and file formats
> the loader assumes.

`fine_tuned_model.pt` (the trained `StressTransformer` weights) **is** included,
so the API and dashboard can be run without the raw recordings.

## Setup

```bash
git clone https://github.com/paulms77/StressAI.git
cd StressAI

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

MySQL is only needed for the training path:

```sql
CREATE DATABASE stress_db CHARACTER SET utf8mb4;
USE stress_db;

CREATE TABLE subjects (
  id INT AUTO_INCREMENT PRIMARY KEY,
  subject_name VARCHAR(32) UNIQUE
);
CREATE TABLE eda_data   (id INT AUTO_INCREMENT PRIMARY KEY, subject_id INT, eda    DOUBLE);
CREATE TABLE bvp_data   (id INT AUTO_INCREMENT PRIMARY KEY, subject_id INT, bvp    DOUBLE);
CREATE TABLE label_data (id INT AUTO_INCREMENT PRIMARY KEY, subject_id INT, labels INT);
```

Configuration is read from the environment:

| Variable | Default | Used by |
| --- | --- | --- |
| `STRESS_MODEL_PATH` | `fine_tuned_model.pt` | API, dashboard |
| `STRESS_RAW_DIR` | `data/Subjects` | preprocessing |
| `STRESS_DATA_DIR` | `data/Subjects_processing` | preprocessing, DB load |
| `DB_HOST` / `DB_PORT` | `127.0.0.1` / `3306` | DB load, training |
| `DB_USER` / `DB_PASSWORD` | `root` / *(empty)* | DB load, training |
| `DB_NAME` | `stress_db` | DB load, training |

## Usage

### Serve the API

```bash
python main.py          # uvicorn on 0.0.0.0:9999, auto-reload
```

```bash
curl -X POST http://localhost:9999/upload_and_predict \
     -F "file=@data/Subjects_processing/subject_02/Subject_data.csv"
```

The uploaded CSV needs `eda` and `bvp` columns. It is cut into 320-sample
(5 second) windows and each window is classified. Interactive docs are at
`http://localhost:9999/docs`.

### Live dashboard

```bash
python dashboard.py     # Dash on http://localhost:8050
```

Replays a recording one second at a time, plotting EDA and BVP alongside the
rolling stress prediction.

### Reproduce from raw recordings

```bash
export STRESS_RAW_DIR=data/Subjects
export STRESS_DATA_DIR=data/Subjects_processing

python stress_data.py   # 1. clean, align, label  → per-subject CSVs
python stress_db.py     # 2. load CSVs into MySQL
python train.py         # 3. train → fine_tuned_model.pt
```

## Repository layout

```
model.py            StressRNN (LSTM) and StressTransformer definitions
stress_data.py      signal processing — interpolation, resampling, filtering,
                    dropout removal, protocol-based labelling
stress_db.py        load per-subject CSVs into MySQL
train.py            sequential per-subject training loop
process.py          small helper for preparing a single subject for inference
main.py             uvicorn entry point
app/app.py          FastAPI service — POST /upload_and_predict
dashboard.py        Dash dashboard — live replay and prediction plots
fine_tuned_model.pt trained StressTransformer weights
data/               expected location of recordings (not included)
```

## Limitations

**Binary labels come from the protocol, not from the participant.** A sample is
labelled "stress" because it fell inside a task block, not because that person
was measurably stressed at that moment. Someone relaxed during a task is
mislabelled, and there is no way to detect that from the signal alone.

**Sequential per-subject fine-tuning is not the same as training on everyone.**
Each subject's model starts from the previous subject's weights, so later
subjects carry more influence and the final weights are biased toward the end of
the sequence. A pooled fit with subject-wise cross-validation would be the
sounder comparison.

**No held-out subject evaluation.** The training loop reports loss only. Honest
numbers for this task require leave-one-subject-out validation, since
within-subject windows are heavily correlated and any random split leaks.

## License

MIT — see [LICENSE](LICENSE).
