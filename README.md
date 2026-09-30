# Machine Learning Experiments

This project runs 15 machine-learning experiments on CIC-IDS traffic data. It combines three datasets, preprocesses the features and labels, applies PCA, trains five models, and saves classification metrics and confusion-matrix reports.

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- XGBoost

Install the required packages with:

```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Input Data

Place these CSV files in the project root, alongside `code.py`:

- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

The script combines these datasets and treats the final column as the target label.

## Run

From the project root, run:

```powershell
python code.py
```

The script uses two PCA components, test sizes of 30%, 50%, and 70%, and these five models: Elastic Net, XGBoost, AdaBoost, Linear Regression, and Multilayer Perceptron. Together, those settings produce 15 experiments.

## Outputs

Results are written to `15_ML_Results/`:

- 15 JPG reports, each with a confusion matrix, metrics, and classification report
- `ALL_15_RESULTS.csv`, containing the summary metrics for all experiments
"# PCA" 
"# PCA" 
