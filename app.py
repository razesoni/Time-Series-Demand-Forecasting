from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Demand Forecasting", layout="wide")


@st.cache_data
def get_options():
    return requests.get(f"{API_URL}/options", timeout=10).json()


@st.cache_data
def get_defaults(store_id, product_id):
    params = {"store_id": store_id, "product_id": product_id}
    return requests.get(f"{API_URL}/defaults", params=params, timeout=10).json()


def get_history(store_id, product_id, end):
    params = {"store_id": store_id, "product_id": product_id, "end": str(end), "days": 60}
    return requests.get(f"{API_URL}/history", params=params, timeout=10).json()


try:
    options = get_options()
except requests.exceptions.ConnectionError:
    st.error("Can't reach the API. Start it first with: uvicorn main:app --reload")
    st.stop()

st.title("Demand Forecasting")
st.write("Pick a store and product, set the details for the day and get the predicted demand.")

store_id = st.sidebar.selectbox("Store", options["stores"])
product_id = st.sidebar.selectbox("Product", options["products"])

defaults = get_defaults(store_id, product_id)
min_date = date.fromisoformat(defaults["min_date"])
max_date = date.fromisoformat(defaults["max_date"])
st.sidebar.caption(f"Dates available: {min_date} to {max_date}")

# keys change with the selection so the form resets to the new defaults
key = f"{store_id}_{product_id}"

with st.form("forecast_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        day = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date, key=f"date_{key}")
        price = st.number_input("Price", min_value=0.0, value=defaults["price"], step=0.5, key=f"price_{key}")
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0,
                                   value=defaults["discount"], step=1.0, key=f"discount_{key}")
        competitor_price = st.number_input("Competitor price", min_value=0.0,
                                           value=defaults["competitor_price"], step=0.5, key=f"comp_{key}")

    with col2:
        category = st.selectbox("Category", options["categories"],
                                index=options["categories"].index(defaults["category"]), key=f"cat_{key}")
        region = st.selectbox("Region", options["regions"],
                              index=options["regions"].index(defaults["region"]), key=f"region_{key}")
        weather = st.selectbox("Weather", options["weather"],
                               index=options["weather"].index(defaults["weather"]), key=f"weather_{key}")
        seasonality = st.selectbox("Season", options["seasonality"],
                                   index=options["seasonality"].index(defaults["seasonality"]), key=f"season_{key}")

    with col3:
        promotion = st.checkbox("Promotion running", value=bool(defaults["promotion"]), key=f"promo_{key}")
        epidemic = st.checkbox("Epidemic", value=bool(defaults["epidemic"]), key=f"epidemic_{key}")

    submitted = st.form_submit_button("Predict demand")

if submitted:
    payload = {
        "store_id": store_id,
        "product_id": product_id,
        "date": str(day),
        "price": price,
        "discount": discount,
        "competitor_price": competitor_price,
        "promotion": int(promotion),
        "epidemic": int(epidemic),
        "category": category,
        "region": region,
        "weather": weather,
        "seasonality": seasonality,
    }

    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    if response.status_code != 200:
        st.error(response.json().get("detail", "Something went wrong"))
        st.stop()

    result = response.json()

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted demand", result["prediction"])
    c2.metric("Last 7 day average", result["last_7_day_avg"])
    if result["actual"] is not None:
        diff = round(result["prediction"] - result["actual"], 1)
        c3.metric("Actual demand", result["actual"], delta=f"{diff} (prediction - actual)", delta_color="off")

    history = pd.DataFrame(get_history(store_id, product_id, day - timedelta(days=1)))
    history["date"] = pd.to_datetime(history["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["date"], history["demand"], color="steelblue", label="Past demand")
    ax.scatter(pd.Timestamp(day), result["prediction"], color="crimson", s=60, zorder=3, label="Prediction")
    if result["actual"] is not None:
        ax.scatter(pd.Timestamp(day), result["actual"], color="green", s=60, zorder=3, label="Actual")
    ax.set_title(f"Store {store_id} - Product {product_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
