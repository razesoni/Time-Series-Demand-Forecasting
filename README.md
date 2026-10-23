# Demand Forecasting App

FastAPI serves the XGBoost model from the notebook, Streamlit is the UI.

## Folder layout

```
demand-app/
  backend/
    main.py
    xgb_demand_model.joblib     <- saved by the notebook (Step 13)
    demand_forecasting.csv      <- same data file the notebook uses
  frontend/
    app.py
  requirements.txt
```

## Run it

```
pip install -r requirements.txt

# terminal 1
cd backend
uvicorn main:app --reload

# terminal 2
cd frontend
streamlit run app.py
```

API docs are at http://localhost:8000/docs

## Notes

- The model needs the last 28 days of demand for lag and rolling features, so the app only
  allows dates from 28 days after the first record up to one day after the last record.
- If the chosen date already exists in the data, the actual demand is shown next to the prediction.
