import glob
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Cab Fare Predictor", page_icon="🚕")

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

@st.cache_resource
def train_model():
    df = pd.read_csv(glob.glob("*.csv")[0])
    df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore").dropna()
    df = df[(df.fare_amount > 0) & (df.fare_amount < 100)]
    df = df[df.passenger_count.between(1, 6)]
    df = df[df.pickup_latitude.between(40, 42) & df.dropoff_latitude.between(40, 42)]
    df = df[df.pickup_longitude.between(-75, -72) & df.dropoff_longitude.between(-75, -72)]
    df["distance_km"] = haversine(df.pickup_latitude, df.pickup_longitude,
                                  df.dropoff_latitude, df.dropoff_longitude)
    df = df[(df.distance_km > 0.1) & (df.distance_km < 100)]
    t = pd.to_datetime(df.pickup_datetime)
    df["hour"], df["day_of_week"] = t.dt.hour, t.dt.dayofweek
    df["month"], df["year"] = t.dt.month, t.dt.year
    feats = ["distance_km", "passenger_count", "hour", "day_of_week", "month", "year"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df.fare_amount, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=40, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, r2_score(y_test, pred), mean_absolute_error(y_test, pred), len(df)

with st.spinner("Training the model on first start (1 to 2 minutes)..."):
    model, r2, mae, rows = train_model()

places = {
    "Times Square": (40.7580, -73.9855),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Central Park": (40.7829, -73.9654),
    "Empire State Building": (40.7484, -73.9857),
    "Wall Street": (40.7060, -74.0088),
}
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.title("🚕 Taxi Fare Prediction")
st.write("Estimate an Uber fare in New York City using a Random Forest model.")
st.caption(f"Trained on {rows:,} rides | R2: {r2:.2f} | Average error: ${mae:.2f}")

pickup = st.selectbox("Pickup location", list(places))
dropoff = st.selectbox("Dropoff location", list(places), index=1)
passengers = st.slider("Passengers", 1, 6, 1)
hour = st.slider("Hour of day (0-23)", 0, 23, 12)
day = st.selectbox("Day of week", days)
month = st.slider("Month", 1, 12, 6)
year = st.slider("Year", 2009, 2015, 2015)

if st.button("Predict fare"):
    d = haversine(*places[pickup], *places[dropoff])
    ride = pd.DataFrame({"distance_km": [d], "passenger_count": [passengers],
                         "hour": [hour], "day_of_week": [days.index(day)],
                         "month": [month], "year": [year]})
    st.metric("Estimated fare", f"${model.predict(ride)[0]:.2f}")
    st.write(f"Distance: {d:.1f} km")
