# Mobile Network Traffic Forecasting in Milan: SARIMAX vs LSTM vs TCN

**Research question:** How do different sequential models compare for one-step-ahead mobile network traffic
forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

- **Data:** Telecom Italia Internet activity, Milan, 1 Nov 2013 – 1 Jan 2014, 10,000 grid squares, 10-minute intervals
  (62 daily files, ~20 GB of raw text).
- **Task:** one-step-ahead (10-minute) forecasting for single grid squares. The test week is 16–22 Dec 2013.
- **Models:** regression with ARMA errors and daily/weekly Fourier terms (SARIMAX), LSTM, and a dilated causal
  convolutional network (TCN). Persistence and seasonal-naive baselines are included for reference.

---

## Main results (test week, MAE)

| Square | Profile | SARIMAX | LSTM | TCN | Persistence |
|---|---|---|---|---|---|
| 5161 (top-1) | leisure, busier at weekends | **77.5** | 80.1 | 78.5 | 92.8 |
| 5059 (top-2) | mixed | **65.8** | 67.2 | 72.9 | 81.5 |
| 5259 (top-3) | office, weekends collapse | 67.9 | **59.0** | 60.3 | 76.0 |
| 4159 | office-like, low volume | **13.8** | 15.5 | 15.2 | 16.0 |
| 4556 | evening-peaked | **25.1** | 28.0 | 27.5 | 28.9 |

- SARIMAX is best or tied in 4 of 5 areas, using 24 parameters against 37,505 (TCN) and 69,249 (LSTM).
- The neural networks win clearly only at square 5259.
- Predicting the **change from the last observation** (residual target), instead of the level, reduced the
  networks' validation RMSE by 17% (LSTM) and 52% (TCN). It also removed large seed-to-seed instability.

Full metric tables (MAE, RMSE, MAPE, WAPE) are in `results/metrics_<square>.csv`. The experiment log is in
`results/experiment_log.csv`.

---

## Repository structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb   # raw files -> compact matrix; memory evidence
│   ├── 02_eda.ipynb                # distribution, spatial map, time-series analyses
│   └── 03_forecasting.ipynb        # tuning, final models, metrics, plots, timing, failure analysis
├── data/                           # small derived files (ODbL 1.0, see data/LICENSE-DATA.md)
│   ├── selected_series.csv         # 10-min series for the 5 analysed squares + city total
│   ├── selected_squares.json       # top-3 and selected square IDs
│   ├── square_totals.csv           # total Internet activity per square
│   ├── processing_log.csv          # per-day processing time, rows, peak memory
│   ├── memory_evidence.json        # naive vs optimised memory measurements
│   └── eda_summary.json            # EDA statistics (distribution, ACF, decomposition, stationarity)
├── results/                        # metrics, predictions, timing, tuning log and decisions
│   └── direct_target_backup/       # results before the residual-target change (round 5)
├── figures/                        # all figures used in the report
├── requirements.txt
└── README.md
```

The notebooks are saved **with their outputs**, so all results can be inspected without re-running anything.

---

## Setup

### Option A: Google Colab (used for all reported results; recommended)
Hardware: Colab runtime with 2 vCPUs, 13.6 GB RAM, and an NVIDIA Tesla T4 GPU (15 GB) for notebook 03.
Software: Python 3.13, TensorFlow 2.20.0, Keras 3.13.2, statsmodels 0.15.0 (see `requirements.txt`).

1. Open each notebook in Colab (File → Open notebook → GitHub tab → paste this repository's URL).
2. The notebooks read and write everything under `MyDrive/milan_forecasting/` in your Google Drive and
   mount Drive automatically. To change this location, edit `PROJECT_DIR` in the first code cell of each notebook.

### Option B: local machine
```bash
pip install -r requirements.txt
```
Replace the `google.colab` Drive-mount lines at the top of each notebook, and set `PROJECT_DIR` to a local folder.

---

## Getting the data

Download the 62 daily files `sms-call-internet-mi-2013-11-01.txt` … `sms-call-internet-mi-2014-01-01.txt`
from Harvard Dataverse: <https://doi.org/10.7910/DVN/EGZHFV>. You can use either of two routes:

- **Drive route:** put the Dataverse zip downloads in `MyDrive/dataverse_files/` (a Drive shortcut to a shared
  folder also works). Notebook 01 reads the files directly from the zips, without extracting them.
- **API route:** do nothing. Any day not found in the zips is downloaded automatically through the Dataverse API
  and checked against its MD5 checksum.

For the spatial map (notebook 02), download the Milano grid GeoJSON from <https://doi.org/10.7910/DVN/QJWLFU>
and save it as `MyDrive/milan_forecasting/processed/milano-grid.geojson`. The Dataverse API blocks requests from
Colab, so this download has to be done manually in a browser.

---

## How to run

Run the notebooks in order with **Runtime → Run all**. All three are **resumable**: finished steps are saved to
Drive and skipped on re-run, so a disconnected session can simply be restarted.

| Notebook | Runtime | Approx. time | Produces |
|---|---|---|---|
| `01_data_preparation` | CPU | 20–40 min | `internet_matrix.npy` (357 MB), memory evidence, `selected_series.csv` |
| `02_eda` | CPU | ~5 min | Figures 1–6, `eda_summary.json` |
| `03_forecasting` | T4 GPU | ~60 min (first run) | Tuning log, predictions, metrics, Figures 7–11, timing |

**Shortcut, without the raw data:** notebook 03 only needs `selected_series.csv` and `selected_squares.json`.
Copy both files from `data/` into `MyDrive/milan_forecasting/processed/`, then run notebook 03.
Notebook 02 needs the full matrix, so it has to be preceded by notebook 01.

**Reproducibility notes**
- Random seeds are fixed (`SEED = 42`). Round 5 also measures variation across seeds 42, 7 and 123.
- GPU training is not bit-for-bit deterministic, so neural-network metrics may vary slightly
  (validation std ≈ 2–4% of MAE).
- The data split is fixed in code: train 1 Nov – 8 Dec, validation 9 – 15 Dec, test 16 – 22 Dec 2013.
  Scaling is fitted on the training period only.

---

## Methodology in brief

- **Memory:** only 3 of 8 columns are read, with compact data types, and activity is summed over country codes
  one day at a time. The result is a dense float32 matrix of 10,000 × 8,928 values (357 MB), compared with an
  estimated 19.2 GB for a naive load.
- **Transform:** `log1p` (daily standard deviation is proportional to daily mean, r = 0.98), followed by
  min-max scaling fitted on the training period.
- **SARIMAX:** ARMA(3,0) errors with a constant; 4 daily and 5 weekly Fourier pairs; a weekend/holiday dummy.
  Test predictions are one-step Kalman-filter predictions with fixed parameters.
- **LSTM:** 36-step window × 6 features (traffic plus 5 calendar features) → LSTM(128) → dropout 0.1 → dense.
- **TCN:** 288-step window, 6 residual blocks of dilated causal convolutions (receptive field 253 steps),
  32 filters, dropout 0.2.
- **Neural network target:** the standardised change from the last observed value (residual target).
- **Tuning:** 5 documented, hypothesis-driven rounds on the validation week (37 logged runs). See
  `results/experiment_log.csv` and `results/tuning_decisions.json`.

---

## Data licence and attribution

Data [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html),
[ODI node Trento](http://theodi.fbk.eu/). The source data and the derived files in `data/` are licensed under the
[Open Database License (ODbL) 1.0](http://opendatacommons.org/licenses/odbl/1-0/).

G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and the Province of Trentino,"
*Sci. Data*, vol. 2, Art. no. 150055, 2015, doi:10.1038/sdata.2015.55.
