import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Header
st.header('LOOKISM')

# Sample images
image = "https://i.pinimg.com/originals/2d/cb/53/2dcb53a89a6e42922f78ef5179d01a11.png"
images = "https://i.pinimg.com/originals/3e/78/46/3e78469701eceb9e014fa70935fbc4ca.jpg"
images_allied = "https://pbs.twimg.com/media/FtmcoFiaYAEQB_g.jpg"

# Sample datasets
df_4_men_crew = pd.DataFrame({
    'Name': ['Jaghyun', 'Seongun', 'Yohan', 'Kim Gimyung'],
    'Age': [19, 20, 21, 20],
    'City': ['Gangnam', 'Gangdong', 'GangBuk', 'Gangseo']
})

df_allied = pd.DataFrame({
    'Name': ['Park HyungSeok', 'Lee Zin', 'Vasco', 'Jay', 'Ahn Hyungseong'],
    'Age': [19, 20, 21, 20, 20]
})

# Sidebar
st.sidebar.title("Menu")
option = st.sidebar.selectbox(
    "Pilih Opsi",
    ("Image", "Dataset", "Grafik", "Prediksi Usia", "Model Regresi")
)

# Main content
if option == "Image":
    st.image(images, caption="Lookism")

# Header and content for 4 MEN CREW
st.header('4 MEN CREW')
if option == "Image":
    st.image(image, caption="4 Men Crew")
elif option == "Dataset":
    st.subheader("Dataset 4 Men Crew")
    st.dataframe(df_4_men_crew)

# Header and content for ALLIED
st.header('ALLIED')
if option == "Image":
    st.image(images_allied, caption="Allied")
elif option == "Dataset":
    st.subheader("Dataset Allied")
    st.dataframe(df_allied)

# Grafik
if option == "Grafik":
    st.subheader("Grafik")

    # Pilihan dataset untuk grafik
    dataset_option = st.selectbox("Pilih Dataset", ["4 Men Crew", "Allied"])
    selected_df = df_4_men_crew if dataset_option == "4 Men Crew" else df_allied

    # Grafik pilihan
    graph_type = st.selectbox("Pilih jenis grafik", ["Bar", "Pie", "Scatter", "Line"])

    if graph_type == "Bar":
        fig, ax = plt.subplots()
        ax.bar(selected_df['Name'], selected_df['Age'], color='skyblue')
        ax.set_xlabel("Nama")
        ax.set_ylabel("Usia")
        ax.set_title(f"Usia Berdasarkan Nama ({dataset_option})")
        st.pyplot(fig)
    elif graph_type == "Pie":
        fig, ax = plt.subplots()
        ax.pie(selected_df['Age'], labels=selected_df['Name'], autopct="%1.1f%%")
        ax.set_title(f"Distribusi Usia Berdasarkan Nama ({dataset_option})")
        st.pyplot(fig)
    elif graph_type == "Scatter":
        fig, ax = plt.subplots()
        ax.scatter(range(len(selected_df['Name'])), selected_df['Age'], color='orange')
        ax.set_xlabel("Index Nama")
        ax.set_ylabel("Usia")
        ax.set_title(f"Usia vs Index ({dataset_option})")
        st.pyplot(fig)
    elif graph_type == "Line":
        fig, ax = plt.subplots()
        ax.plot(selected_df['Name'], selected_df['Age'], marker='o', color='b')
        ax.set_xlabel("Nama")
        ax.set_ylabel("Usia")
        ax.set_title(f"Perubahan Usia Berdasarkan Nama ({dataset_option})")
        st.pyplot(fig)

# Prediksi Usia
elif option == "Prediksi Usia":
    st.subheader("Prediksi Usia")

    # Simple linear regression model
    X = np.array([18, 19, 20, 21, 22]).reshape(-1, 1)
    y = np.array([18, 19, 20, 21, 22])
    model = LinearRegression()
    model.fit(X, y)

    # User input for prediction
    year = st.slider("Masukkan tahun kelahiran (perkiraan)", 2000, 2024, 2020)
    try:
        pred_age = model.predict([[2024 - year]])
        st.write(f"Prediksi usia Anda adalah **{pred_age[0]:.2f} tahun**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

    # Plotting the linear regression model
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Actual')
    ax.plot(X, model.predict(X), color='red', label='Model Prediksi')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Usia')
    ax.set_title('Model Regresi Linear Prediksi Usia')
    ax.legend()
    st.pyplot(fig)

# Model Regresi
elif option == "Model Regresi":
    st.subheader("Visualisasi Model Regresi")

    X = np.array([18, 19, 20, 21, 22]).reshape(-1, 1)
    y = np.array([18, 19, 20, 21, 22])
    model = LinearRegression()
    model.fit(X, y)

    year = st.slider("Masukkan tahun kelahiran (perkiraan)", 2000, 2024, 2020)
    try:
        pred_age = model.predict([[2024 - year]])
        st.write(f"Prediksi usia Anda adalah **{pred_age[0]:.2f} tahun**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Actual')
    ax.plot(X, model.predict(X), color='red', label='Model Prediksi')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Usia')
    ax.set_title('Model Regresi Linear Prediksi Usia')
    ax.legend()
    st.pyplot(fig)
