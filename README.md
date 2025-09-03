# Wind Turbine Ice Detection System

End-to-end ML pipeline for **icing risk (binary)** and **turbine operating state (multiclass)** classification.  
Built with **scikit-learn** and **XGBoost**, includes **SHAP** explainability and a **Streamlit** app for interactive exploration.  
Large datasets and trained models are intentionally excluded so the repo stays lightweight and reproducible.

---

## 🔑 Key Features

- 🧠 **Binary & Multiclass models**: Logistic Regression and XGBoost baselines
- 📊 **Evaluation utilities**: metrics, confusion matrices, model comparison
- 🧩 **Explainability**: SHAP (global + local attributions)
- 🖥️ **Streamlit dashboard**: quick what-if analysis and visualization
- 🧪 **Reproducible scripts**: one-command training/evaluation
- 💾 **Data/model hygiene**: CSVs and pickled models are ignored by default

---

## 📦 Requirements

- Python 3.9+ recommended
- Install Python deps:
  ```bash
  pip install -r requirements.txt
