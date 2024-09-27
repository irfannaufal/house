import numpy as np
import pandas as pd
import streamlit as st
import pickle
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Load the model 
with open('house_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("California Housing Prediction App")
st.sidebar.header("Input Features")

# User inputs
housing_median_age = st.sidebar.slider('Average age of house (years)', 0.0, 200.0, 50.0)
median_income = st.sidebar.number_input('Median income (in millions)', 0.00, 20.00, 3.87)
population = st.sidebar.slider('Population in the vicinity (millions)', 50.00, 20000.00, 1500.00)
households = st.sidebar.slider('Number of households', 0, 7000, 500)
total_rooms = st.sidebar.slider('Total room area', 0, 50000, 2000)
total_bedrooms = st.sidebar.slider('Total bedroom area', 0, 50000, 500)
lat = st.sidebar.number_input("Latitude", 32.5121, 42.0126, 37.3, step=0.1)
long = st.sidebar.number_input("Longitude", -124.6509, -114.1315, -122.37, step=0.1)

if st.button('Estimate Price'):
    # Prepare input features
    input_features = np.array([housing_median_age, total_rooms, total_bedrooms, population, 
                               households, median_income, lat, long]).reshape(1, -1)

    # Make prediction
    with st.spinner("Calculating..."):
        predicted_price = model.predict(input_features)

    st.success(f"Estimated House Price: ${predicted_price[0]:,.2f}")

    # Create a map
    map_center = [lat, long]
    house_map = folium.Map(location=map_center, zoom_start=12)

    # Add a marker for the estimated house location
    folium.Marker(location=map_center, 
                  popup=f"Estimated Price: ${predicted_price[0]:,.2f}", 
                  icon=folium.Icon(color='blue')).add_to(house_map)

    st_folium(house_map, width=725)


