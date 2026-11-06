# Data layout

The biosignal recordings are **not included in this repository** — they are
physiological measurements from human participants. The code expects them here,
in the structure below.

## Raw recordings — `data/Subjects/`

Empatica E4 session exports, one folder per participant
(`subject_01` … `subject_29`). Each CSV is headerless, one value per line, in
recording order.

```
data/Subjects/
└── subject_01/
    ├── ACC.csv     3 columns (x, y, z), 32 Hz
    ├── BVP.csv     1 column, 64 Hz   ← used
    ├── EDA.csv     1 column,  4 Hz   ← used
    ├── HR.csv      1 column,  1 Hz
    ├── IBI.csv     2 columns (time, interval), event-driven
    └── TEMP.csv    1 column,  4 Hz
```

Only `BVP.csv` and `EDA.csv` are read by the current pipeline.

## Processed — `data/Subjects_processing/`

Written by `stress_data.py`, one folder per participant:

```
data/Subjects_processing/
└── subject_01/
    └── Subject_data.csv     columns: eda, bvp, labels
```

Both channels are at 64 Hz here (EDA upsampled from 4 Hz), and `labels` is
`0` for rest blocks and `1` for task blocks.

## Recording protocol

Alternating rest and task blocks, durations in minutes:

| Rest 1 | Task 1 | Rest 2 | Task 2 | Rest 3 | Task 3 | Rest 4 | Task 4 | Rest 5 | Task 5 | Rest 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 10 | 2 | 5 | 2 | 3 | 2 | — | 2 | 1 | 2 |

## Using your own data

Any source works as long as the processed CSV has `eda`, `bvp` and `labels`
columns at a common sampling rate. Set `STRESS_RAW_DIR` and `STRESS_DATA_DIR` to
point at it.
