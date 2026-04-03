import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title='Sales Prediction',
    page_icon="📈"
)

st.write("# Bigmart Sales Analysis! 🛒📈📊")

st.write("""
    This application allows you to predict the sales of items in retail outlets based on various input features.
    
    **Main Features:**
    1. **Sales Prediction**
    2. **Data Visualizations**
    3. **Stock Prediction**
""")

tab1, tab2 = st.tabs(['Predict', 'Data Visualization'])

with tab1:
    try:
        reg = joblib.load('Ridge_Regression_best_model (4).pkl')  # ✅ FIXED
        clean_df = pd.read_csv('cleaned_data.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    st.title('Sales Prediction App')

    st.header("Item Details")
    item_id = st.selectbox('Item ID', clean_df['Item_Identifier'].unique())
    weight = st.number_input('Weight (in kg)',
                             min_value=float(clean_df['Item_Weight'].min()),
                             max_value=float(clean_df['Item_Weight'].max()))

    fat_content = st.selectbox('Fat Content', clean_df['Item_Fat_Content'].unique())

    visibility = st.number_input('Visibility (percentage)',
                                 min_value=float(clean_df['Item_Visibility'].min()),
                                 max_value=float(clean_df['Item_Visibility'].max()))

    item_type = st.selectbox('Item Type', clean_df['Item_Type'].unique())

    item_price = st.number_input('Item Price',
                                 min_value=float(clean_df['Item_MRP'].min()),
                                 max_value=float(clean_df['Item_MRP'].max()))

    st.header("Outlet Details")
    outlet_id = st.selectbox('Outlet ID', clean_df['Outlet_Identifier'].unique())

    year = st.number_input('Year of Establishment',
                           min_value=int(clean_df['Outlet_Establishment_Year'].min()),
                           max_value=int(clean_df['Outlet_Establishment_Year'].max()))

    size = st.selectbox('Outlet Size', clean_df['Outlet_Size'].unique())
    location = st.selectbox('Outlet Location Type', clean_df['Outlet_Location_Type'].unique())
    outlet_type = st.selectbox('Outlet Type', clean_df['Outlet_Type'].unique())

    if st.button('Predict Sales'):
        new_data = pd.DataFrame({
            'Item_Identifier': [item_id],
            'Item_Weight': [weight],
            'Item_Fat_Content': [fat_content],
            'Item_Visibility': [visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_price],
            'Outlet_Identifier': [outlet_id],
            'Outlet_Establishment_Year': [year],
            'Outlet_Size': [size],
            'Outlet_Location_Type': [location],
            'Outlet_Type': [outlet_type]
        })

        try:
            pred = reg.predict(new_data)
            price = np.exp(pred)

            yearly_price = price[0] * 12
            monthly_price = yearly_price / 12

            predicted_total_stock = yearly_price // item_price if item_price > 0 else 0
            predicted_monthly_stock = predicted_total_stock // 11

            st.success(f"Predicted Sales: ₹{price[0].round(2)}")
            st.info(f"Monthly Sales: ₹{monthly_price.round(2)}")
            st.info(f"Yearly Sales: ₹{yearly_price.round(2)}")
            st.info(f"Monthly Stock: {round(float(predicted_monthly_stock))}")
            st.info(f"Yearly Stock: {predicted_total_stock.round(0)}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

with tab2:
    df = pd.read_csv('cleaned_data.csv')
    st.title('Data Visualization')

    choice = st.selectbox("Choose Visualization:",
                          ['Price vs Sales', 'Outlet Type Sales'])

    if choice == 'Price vs Sales':
        fig = px.scatter(df, x='Item_MRP', y='Item_Outlet_Sales')
        st.plotly_chart(fig)

    elif choice == 'Outlet Type Sales':
        data = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().reset_index()
        fig = px.bar(data, x='Outlet_Type', y='Item_Outlet_Sales')
        st.plotly_chart(fig)
