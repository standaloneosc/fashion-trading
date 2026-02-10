# Designer resale — model & simulation

Pipeline for **feature engineering**, **logistic and survival models**, an **inventory-aware simulator** with Monte Carlo rollouts, and a **LaTeX note** (`docs/research_report.tex`).

**Stack:** Python · NumPy · Pandas · scikit-learn · lifelines · matplotlib.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place listing and sold-history CSVs under data/ (see src/collect/io.py for expected names)
python3 run_pipeline.py
```

Optional env vars (see `.env.example`): `BACKTEST_ROLLOUTS`, `SIMULATOR_DEMAND_MULTIPLIER`, inventory caps, etc.

## Report

Source: `docs/research_report.tex`. A compiled copy is tracked as **`docs/fashion_trading.pdf`**.

To rebuild from source (two passes):

```bash
cd docs && pdflatex research_report.tex && pdflatex research_report.tex
```

Figures in the note are TikZ; reported numbers are tied to whatever CSV inputs you last ran through the pipeline.

## Layout

| Path | Role |
|------|------|
| `run_pipeline.py` | End-to-end pipeline |
| `src/features/`, `src/models/` | Signals and estimators |
| `src/strategies/`, `src/backtest/` | Policies and simulator |
| `docs/research_report.tex` | Research write-up (LaTeX) |
| `docs/fashion_trading.pdf` | Compiled research PDF |

Generated at runtime (not in git): `data/*.csv`, `reports/*`, extra LaTeX build artifacts under `docs/*.aux` etc.
