import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import io
from geopy.distance import geodesic

def time_to_minutes(time_str):
    if pd.isna(time_str) or time_str == 'NaN':
        return np.nan
    try:
        t = datetime.strptime(time_str, '%H:%M:%S')
        return t.hour * 60 + t.minute
    except:
        return np.nan

# Preprocessing function
def preprocess_data(df):
    # Handle missing values
    numeric_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
    for col in numeric_cols:
        df[col] = df[col].replace('NaN', np.nan).astype(float, errors='ignore')
        df[col].fillna(df[col].median(), inplace=True)

    # Calculate distance
    df['distance_km'] = df.apply(lambda row: geodesic((row["Restaurant_latitude"], row["Restaurant_longitude"]),
                                                       (row["Delivery_location_latitude"], row["Delivery_location_longitude"])).kilometers, axis=1)

   
    # Fix negative latitudes
    df['Restaurant_latitude'] = df['Restaurant_latitude'].abs()
    df['Delivery_location_latitude'] = df['Delivery_location_latitude'].abs()

    # Time features
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['day_of_week'] = df['Order_Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
   
    df['order_time_min'] = df['Time_Orderd'].apply(time_to_minutes)
    df['pickup_time_min'] = df['Time_Order_picked'].apply(time_to_minutes)
    df['prep_time'] = df['pickup_time_min'] - df['order_time_min']
    df['prep_time'].fillna(df['prep_time'].median(), inplace=True)

    # Clean categorical variables (simple encoding since no LabelEncoders provided)
    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '').replace('NaN', 'Unknown')
    df['Road_traffic_density'] = df['Road_traffic_density'].replace('NaN', 'Unknown')

    # Manual encoding (assuming model expects numeric inputs)
    weather_map = {'Sunny': 0, 'Stormy': 1, 'Sandstorms': 2, 'Cloudy': 3, 'Fog': 4, 'Windy': 5, 'Unknown': 6}
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3, 'Unknown': 4}
    order_map = {'Snack': 0, 'Drinks': 1, 'Buffet': 2, 'Meal': 3}
    vehicle_map = {'motorcycle': 0, 'scooter': 1, 'electric_scooter': 2, 'bicycle': 3}
    festival_map = {'No': 0, 'Yes': 1}
    city_map = {'Metropolitian': 0, 'Urban': 1, 'Semi-Urban': 2, 'NaN': 3}

    df['Weatherconditions'] = df['Weatherconditions'].map(weather_map).fillna(6)  # Default to Unknown
    df['Road_traffic_density'] = df['Road_traffic_density'].map(traffic_map).fillna(4)
    df['Type_of_order'] = df['Type_of_order'].map(order_map)
    df['Type_of_vehicle'] = df['Type_of_vehicle'].map(vehicle_map)
    df['Festival'] = df['Festival'].map(festival_map).fillna(0)  # Default to No
    df['City'] = df['City'].map(city_map).fillna(3)  # Default to NaN encoded as 3
   
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
               'Weatherconditions', 'Road_traffic_density', 'Vehicle_condition',
               'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries',
               'Festival', 'City', 'day_of_week', 'is_weekend', 'prep_time']
   
    X = df[features]
    return X, df



# Set theme colors in sidebar
st.sidebar.image("https://media-hosting.imagekit.io//e09e1605348e4c6c/logo.webp?Expires=1834682834&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=DH8-nP--gUEWM~PjiJwqKBZcX4WL77xHEK5ZbByTlHVOQb0pkj0AWCel~8B5~eBU3Z5K9rYeDipVEff-4DVLXsY~2ohp39BM9NxIOuEfglPLijt3PhfafxrUJqyoMvPdYNp~teOq0t~pZCB2pxqwwAjHXf-ViyEF6l~46CuCsJwXdxc1pEZoCwYknTeIpN7OQsNXdX9Hup5nb6JfVNAlmqyWbMVtH~BvLRYaw7S5Aj7N6yyV5Ykp1xoNEmJ6XfaQmWtzApFF8Hoeq6FPImpt2bs3DezbDGECFCTlZTcEqKPR2wbUTBH8ccVa0DNco4CqBNAYtvlLD0hPsLViIoc4Zw__", width=150)
st.sidebar.header("🚀 Navigation")
menu = st.sidebar.radio("📌 Select an option", ["🏠 Home", "📊 Predictions", "📝 About"])

if menu == "🏠 Home":
    st.title("🍕 Food Delivery Time Prediction")
    # st.subheader("Estimate delivery time based on real-time conditions!")
    # st.image("delivery-icon.png", width=300)
    st.write("Upload your dataset or manually enter delivery details to get a prediction.")
    st.write("""
    **Features:**
    - Predict delivery time using Machine Learning 📊
    - Analyze traffic, weather, and order type impact ☁️🚦
    - Download predictions for further analysis 📥
    """)

elif menu == "📊 Predictions":
    st.title("🚀 Predict Delivery Time")
    option = st.sidebar.selectbox("Choose an option", ["Upload CSV", "Manual Input"])
    
    if option == "Upload CSV":
        st.header("Upload Test CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Preprocess and predict (Dummy model loading here)
            try:
                rf_model = joblib.load('lr_model.pkl')
                X, df_processed = preprocess_data(df)
                predictions = rf_model.predict(X)
                df_processed['Predicted_Time_taken(min)'] = predictions
                
                st.write("Predictions:")
                st.dataframe(df_processed[['ID', 'Predicted_Time_taken(min)']])
                
                csv = df_processed.to_csv(index=False)
                st.download_button("Download Predictions", data=csv, file_name="test_predictions.csv", mime="text/csv")
            except FileNotFoundError:
                st.error("Model file not found. Ensure it is in the same directory.")
    
    elif option == "Manual Input":
        st.header("Enter Delivery Details")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Delivery Person Age", min_value=18, max_value=60, value=30)
            ratings = st.number_input("Delivery Person Ratings", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            rest_lat = st.number_input("Restaurant Latitude", value=12.914264)
            rest_lon = st.number_input("Restaurant Longitude", value=77.6784)
            del_lat = st.number_input("Delivery Latitude", value=12.924264)
            del_lon = st.number_input("Delivery Longitude", value=77.6884)
        
        with col2:
            order_date = st.date_input("Order Date")
            time_order = st.text_input("Time Ordered (HH:MM:SS)", "08:30:00")
            time_picked = st.text_input("Time Picked (HH:MM:SS)", "08:45:00")
            weather = st.selectbox("Weather Conditions", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy", "Unknown"])
            traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam", "Unknown"])
            vehicle_cond = st.number_input("Vehicle Condition", min_value=0, max_value=3, value=1)
            order_type = st.selectbox("Type of Order", ["Snack", "Drinks", "Buffet", "Meal"])
            vehicle_type = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
            multi_del = st.number_input("Multiple Deliveries", min_value=0, max_value=5, value=1)
            festival = st.selectbox("Festival", ["No", "Yes"])
            city = st.selectbox("City", ["Metropolitian", "Urban", "Semi-Urban", "NaN"])
        
        if st.button("Predict"):
            with st.spinner("🚴‍♂️ Estimating delivery time..."):
                try:
                    rf_model = joblib.load('random_forest.pkl')
                    input_data = pd.DataFrame({
                        'Delivery_person_Age': [age],
                        'Delivery_person_Ratings': [ratings],
                        'Restaurant_latitude': [rest_lat],
                        'Restaurant_longitude': [rest_lon],
                        'Delivery_location_latitude': [del_lat],
                        'Delivery_location_longitude': [del_lon],
                        'Order_Date': [order_date.strftime('%d-%m-%Y')],
                        'Time_Orderd': [time_order],
                        'Time_Order_picked': [time_picked],
                        'Weatherconditions': [weather],
                        'Road_traffic_density': [traffic],
                        'Vehicle_condition': [vehicle_cond],
                        'Type_of_order': [order_type],
                        'Type_of_vehicle': [vehicle_type],
                        'multiple_deliveries': [multi_del],
                        'Festival': [festival],
                        'City': [city]                    })
                    
                    X, _ = preprocess_data(input_data)
                    prediction = rf_model.predict(X)[0]
                    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes ⏳")
                except FileNotFoundError:
                    st.error("Model file not found. Ensure it is in the same directory.")
elif menu == "📝 About":
    st.title("📝 About")
    st.write("App Name: Delfast Delivery Time Predictor")
    st.write("Developer: Aswathy S Udayan")
    st.write("Version: 1.0")
    st.write("Purpose: This application predicts food delivery times based on various factors like traffic, weather, and order details using Machine Learning")

