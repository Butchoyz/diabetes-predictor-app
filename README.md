# diabetes-predictor-app
Streamlit app for predicting diabetes using a stacked ensemble model.

## Checking the SHAP numbers

The app computes exact SHAP values directly from the thesis formulas (base risk, Shapley value,
additive feature attribution) and shows every step. To confirm they match the `shap` library and
add up to each model's predicted probability, including the meta-learner breakdown, run:

```bash
python verify_shap.py
```
