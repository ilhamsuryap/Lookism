import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt

# Load model prediksi
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Judul aplikasi
st.title('Prediksi Harga Mobil')
st.header("Dataset")

# Load file CSV dataset
df1 = pd.read_csv('CarPrice.csv')
st.dataframe(df1)

# Menampilkan grafik Highway-mpg
st.write("Grafik Highway-mpg")
chart_highwaympg = pd.DataFrame(df1, columns=["highway-mpg"])
st.line_chart(chart_highwaympg)

# Menampilkan grafik curbweight
st.write("Grafik curbweight")
chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
st.line_chart(chart_curbweight)

# Menampilkan grafik horsepower
st.write("Grafik horsepower")
chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
st.line_chart(chart_horsepower)

# Input nilai dari variabel independent
st.sidebar.header("Input Data")
highwaympg = st.sidebar.number_input("Masukkan nilai highway-mpg", min_value=0.0, max_value=500.0, step=0.1)
curbweight = st.sidebar.number_input("Masukkan nilai curbweight", min_value=0.0, max_value=5000.0, step=0.1)
horsepower = st.sidebar.number_input("Masukkan nilai horsepower", min_value=0.0, max_value=500.0, step=0.1)

# Tombol prediksi
if st.button('Prediksi'):
    try:
        # Prediksi harga mobil
        # Pastikan urutan input sesuai dengan model yang dilatih
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]]) 

        # Format hasil prediksi
        harga_mobil_float = float(car_prediction[0])
        harga_mobil_formated = f"Rp {harga_mobil_float:,.2f}"

        st.success(f"Harga Mobil yang Diprediksi: {harga_mobil_formated}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")