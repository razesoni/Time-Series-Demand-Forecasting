from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Demand Forecasting API")

# model and data are loaded once when the server starts
saved = joblib.load(BASE_DIR / "xgb_demand_model.joblib")
model = saved["model"]
feature_names = saved["features"]

df = pd.read_csv(BASE_DIR / "demand_forecasting.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Store ID"] = df["Store ID"].astype(str)
df["Product ID"] = df["Product ID"].astype(str)
df = df.sort_values("Date")

CAT_COLS = ["Category", "Region", "Weather Condition", "Seasonality"]


class PredictRequest(BaseModel):
    store_id: str
    product_id: str
    date: date
    price: float
    discount: float
    competitor_price: float
    promotion: int
    epidemic: int
    category: str
    region: str
    weather: str
    seasonality: str


def get_series(store_id, product_id):
    rows = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
    if rows.empty:
        raise HTTPException(404, "No data for this store and product")
    return rows.set_index("Date")["Demand"]


@app.get("/options")
def options():
    return {
        "stores": sorted(df["Store ID"].unique().tolist()),
        "products": sorted(df["Product ID"].unique().tolist()),
        "categories": sorted(df["Category"].unique().tolist()),
        "regions": sorted(df["Region"].unique().tolist()),
        "weather": sorted(df["Weather Condition"].unique().tolist()),
        "seasonality": sorted(df["Seasonality"].unique().tolist()),
    }


@app.get("/defaults")
def defaults(store_id: str, product_id: str):
    # latest row for this store + product, used to pre-fill the form
    series = get_series(store_id, product_id)
    last = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].iloc[-1]

    return {
        "category": last["Category"],
        "region": last["Region"],
        "weather": last["Weather Condition"],
        "seasonality": last["Seasonality"],
        "price": float(last["Price"]),
        "discount": float(last["Discount"]),
        "competitor_price": float(last["Competitor Pricing"]),
        "promotion": int(last["Promotion"]),
        "epidemic": int(last["Epidemic"]),
        # the model needs 28 days of past demand, so the first usable date is 28 days in
        "min_date": str((series.index.min() + pd.Timedelta(days=28)).date()),
        "max_date": str((series.index.max() + pd.Timedelta(days=1)).date()),
    }


@app.get("/history")
def history(store_id: str, product_id: str, end: date, days: int = 60):
    series = get_series(store_id, product_id)
    end = pd.Timestamp(end)
    recent = series[(series.index <= end) & (series.index > end - pd.Timedelta(days=days))]
    return [{"date": str(d.date()), "demand": float(v)} for d, v in recent.items()]


@app.post("/predict")
def predict(req: PredictRequest):
    series = get_series(req.store_id, req.product_id)
    day = pd.Timestamp(req.date)

    # same lag and rolling features as the notebook, built from past demand only
    past = series.reindex(pd.date_range(day - pd.Timedelta(days=28), day - pd.Timedelta(days=1)))
    if past.isna().any():
        raise HTTPException(
            400, "Not enough past demand for this date. Pick a date inside the allowed range."
        )

    last_7 = past.tail(7)
    lags = {f"lag_{n}": float(past.iloc[-n]) for n in [1, 7, 14, 28]}

    row = {
        "Price": req.price,
        "Discount": req.discount,
        "Discounted Price": req.price * (1 - req.discount / 100),
        "Competitor Pricing": req.competitor_price,
        "Price Gap": req.price - req.competitor_price,
        "Promotion": req.promotion,
        "Epidemic": req.epidemic,
        "Month": day.month,
        "Day": day.day,
        "DayOfWeek": day.dayofweek,
        "WeekOfYear": int(day.isocalendar().week),
        "Quarter": day.quarter,
        "IsWeekend": int(day.dayofweek >= 5),
        **lags,
        "roll_mean_7": float(last_7.mean()),
        "roll_mean_28": float(past.mean()),
        "roll_std_7": float(last_7.std()),
        "Category": req.category,
        "Region": req.region,
        "Weather Condition": req.weather,
        "Seasonality": req.seasonality,
    }

    X = pd.get_dummies(pd.DataFrame([row]), columns=CAT_COLS, dtype=int)
    X = X.reindex(columns=feature_names, fill_value=0)

    prediction = float(model.predict(X)[0])
    actual = float(series[day]) if day in series.index else None

    return {
        "prediction": round(max(prediction, 0), 1),
        "actual": actual,
        "last_7_day_avg": round(float(last_7.mean()), 1),
    }
