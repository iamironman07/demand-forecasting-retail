"""Hybrid STL + SARIMA + LightGBM demand forecasting with temporal CV and SHAP."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

PERIOD = 52
HORIZONS = [4, 8, 12]
FEATURES = ["promo", "price", "holiday", "week", "month", "seasonal", "t"]


def load(path="sales.csv"):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["t"] = np.arange(len(df))
    return df


def fit_predict(train, test):
    """Return (stl_sarima_forecast, hybrid_forecast) for the test window."""
    h = len(test)
    stl = STL(train["sales"].values, period=PERIOD, robust=True).fit()
    seasonal_hist = stl.seasonal
    deseason = train["sales"].values - seasonal_hist

    sar = SARIMAX(deseason, order=(1, 1, 1), trend="t").fit(disp=False)
    base_future = sar.forecast(h) + seasonal_hist[-PERIOD:][:h]

    base_train = sar.fittedvalues + seasonal_hist
    resid = train["sales"].values - base_train

    tr = train.assign(seasonal=seasonal_hist)
    te = test.assign(seasonal=seasonal_hist[-PERIOD:][:h])
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=5, verbose=-1)
    model.fit(tr[FEATURES].iloc[PERIOD:], resid[PERIOD:])
    hybrid = base_future + model.predict(te[FEATURES])
    return base_future, hybrid, model, tr


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)),
            "MAPE%": np.mean(np.abs((y - p) / y)) * 100}


def temporal_cv(df, n_folds=4):
    rows = []
    for h in HORIZONS:
        for k in range(n_folds, 0, -1):
            cut = len(df) - k * h
            train, test = df.iloc[:cut], df.iloc[cut:cut + h]
            base, hyb, *_ = fit_predict(train, test)
            for name, p in [("STL+SARIMA", base), ("Hybrid", hyb)]:
                rows.append({"horizon": h, "model": name, **metrics(test["sales"], p)})
    res = pd.DataFrame(rows).groupby(["horizon", "model"]).mean().round(2)
    return res


if __name__ == "__main__":
    df = load()
    print("Temporal cross-validation (expanding window):\n")
    res = temporal_cv(df)
    print(res, "\n")
    res.to_csv("cv_results.csv")

    # Final model + SHAP on the last 12 weeks hold-out
    train, test = df.iloc[:-12], df.iloc[-12:]
    base, hyb, model, tr = fit_predict(train, test)
    plt.figure(figsize=(9, 4))
    plt.plot(df["date"].iloc[-60:], df["sales"].iloc[-60:], label="Actual")
    plt.plot(test["date"], base, "--", label="STL+SARIMA")
    plt.plot(test["date"], hyb, label="Hybrid")
    plt.legend(); plt.title("12-week forecast"); plt.tight_layout()
    plt.savefig("forecast.png", dpi=120)

    X = tr[FEATURES].iloc[PERIOD:]
    shap_values = shap.TreeExplainer(model).shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout(); plt.savefig("shap_summary.png", dpi=120)
    print("Saved forecast.png, shap_summary.png, cv_results.csv")
