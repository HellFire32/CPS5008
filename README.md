# Customer churn — CPS5008 ML artefact

Predicts **customer churn** from `data/customer_account_and_usage.csv` using a reproducible scikit-learn pipeline: baseline dummy model, **logistic regression**, **random forest**, and **histogram gradient boosting**, with stratified CV and hyperparameter search. The **champion** is chosen by **validation PR-AUC** (test labels are not used for selection). **Decision thresholds** are tuned on the validation split by minimising **expected cost** (`BUSINESS_COST_FP` / `BUSINESS_COST_FN` in `churn/config.py`). See **`docs/BUSINESS_FRAMING.md`** for report-ready wording on task definition and FP/FN.

## Setup

```bash
cd /path/to/CustomerChurn   # or your clone path
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

From the repository root (this folder):

```bash
PYTHONPATH=. python -m churn.train
```

## Figures (for the report)

After training (so `artifacts/` and `reports/` exist):

```bash
PYTHONPATH=. python -m churn.plots
```

Writes PNGs under `reports/figures/`:

- `model_comparison_test.png` — ROC-AUC and PR-AUC by model  
- `confusion_matrix_champion.png` — champion confusion matrix  
- `segment_performance.png` — ROC-AUC and PR-AUC by region, gender, tariff  
- `permutation_importance_top15.png` — top features with error bars  
- `fairness_roc_auc_spread.png` — segment ROC-AUC spread  

Outputs:

- `artifacts/metrics.json` — CV scores, test metrics, champion summary
- `artifacts/champion_pipeline.joblib` — fitted pipeline + decision threshold
- `reports/segment_metrics.csv` — performance by region, gender, tariff
- `reports/permutation_importance.csv` — feature-level importance (PR-AUC)

## Dependencies

Python 3.10+ recommended. See `requirements.txt`.
