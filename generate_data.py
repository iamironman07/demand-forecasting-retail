"""Generate a synthetic weekly retail sales dataset (4 years)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
weeks = pd.date_range("2021-01-03", periods=208, freq="W")
t = np.arange(len(weeks))
promo = (rng.random(len(weeks)) < 0.15).astype(int)
price = 100 + rng.normal(0, 2, len(weeks)) - 8 * promo
holiday = ((weeks.month == 12) & (weeks.day > 15)).astype(int)
sales = (500 + 1.2 * t                                   # trend
         + 80 * np.sin(2 * np.pi * t / 52)               # yearly seasonality
         + 120 * promo + 150 * holiday - 2.5 * (price - 100)
         + rng.normal(0, 20, len(weeks)))
pd.DataFrame({"date": weeks, "sales": sales.round(1), "promo": promo,
              "price": price.round(2), "holiday": holiday}).to_csv("sales.csv", index=False)
print("sales.csv created:", len(weeks), "rows")
