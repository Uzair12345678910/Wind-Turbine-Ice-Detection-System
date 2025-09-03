# Wind Turbine Ice Detection System

Supervised ML pipeline for **icing risk** and **multiclass turbine states**
using scikit-learn and XGBoost. Includes data prep, model training,
SHAP analysis, and an interactive **Streamlit** app.

## Quick start
pip install -r requirements.txt

## Train (examples)
python train_ice_risk_model.py
python train_ice_risk_xgboost.py
python train_multiclass_logistic_regression_model.py
python train_multiclass_xgboost_model.py

## Evaluate / Explain
python multiclass_model_comparison.py
python confusion_matrix_plot.py
python shap_analysis_multiclass_models.py

## Run the app
streamlit run streamlit_app.py

> Large data (`*.csv`) and trained models (`*.pkl`) are ignored by default.
> Recreate them with the scripts above or add small samples.
