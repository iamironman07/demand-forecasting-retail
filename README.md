# Demand Forecasting in Retail

Hybrid weekly sales forecasting workflow:

1. **STL** decomposition extracts the yearly seasonal component.
2. **SARIMA** models the deseasonalised series (trend + noise).
3. **LightGBM** learns the remaining residuals from promo, price, holiday and calendar features.
4. **Temporal (expanding-window) cross-validation** across 4/8/12-week horizons.
5. **SHAP** explains which features drive demand, to support inventory planning.

## Run
```bash
pip install -r requirements.txt
python generate_data.py      # synthetic data (replace sales.csv with real data)
python forecast.py           # CV metrics + forecast.png + shap_summary.png
```
Tech: Python, pandas, statsmodels, LightGBM, SHAP.
