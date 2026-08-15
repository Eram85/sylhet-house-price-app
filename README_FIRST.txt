THESIS 3.0 - FULL RESTORED + ENHANCED VERSION

IMPORTANT:
This package keeps the ORIGINAL thesis3.0 model.py and app.py structure and functionality.
The earlier shortened version has NOT been used here.

Files:
- model.py  : full original pipeline plus additional evaluation/error-analysis/SHAP/OOD artifacts
- app.py    : full original Streamlit design plus scientifically safer explanations
- eda.py
- sylhet_real_estate.csv
- requirements.txt

FIRST RUN:
1. Open this folder in VS Code.
2. Install dependencies:
   python3 -m pip install -r requirements.txt
3. Retrain/regenerate artifacts with:
   python3 model.py
   (This can take several minutes because it performs 5-fold CV and hyperparameter tuning.)
4. After model.py finishes:
   streamlit run app.py

The baseline mean-price predictor is intentionally included as a reference model.
A negative baseline R2 is valid and is not an error.

The final model is selected by highest independent-test R2. MAE, RMSE, MAPE,
MedianAE, adjusted R2, and cross-validation stability are also reported.
