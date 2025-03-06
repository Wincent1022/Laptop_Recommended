import streamlit as st
import pandas as pd
import joblib

# Load the trained recommendation model
def load_model():
    df, similarity_matrix = joblib.load("laptop_recommendation_model.joblib")
    return df, similarity_matrix

df, similarity_matrix = load_model()

# Function to get laptop recommendations
def get_recommendations(laptop_index, num_recommendations=5):
    scores = list(enumerate(similarity_matrix[laptop_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in scores[1:num_recommendations+1]]
    return df.iloc[recommended_indices]

# Streamlit UI
st.title("Laptop Recommendation System")
st.write("Select your preferences to find the best laptop recommendations.")

# User selection filters
selected_brand = st.selectbox("Select a Brand", df['brand'].unique())
selected_processor = st.selectbox("Select Processor Brand", df['processor_brand'].unique())
selected_ram = st.selectbox("Select RAM Size (GB)", sorted(df['ram_memory'].unique()))
selected_gpu = st.selectbox("Select GPU Brand", df['gpu_brand'].unique())
price_range = st.slider("Select Price Range (USD)", float(df['Price'].min()), float(df['Price'].max()),
                        (float(df['Price'].min()), float(df['Price'].max())))

# Filter laptops based on user selection
filtered_df = df[(df['brand'] == selected_brand) & (df['processor_brand'] == selected_processor) &
                 (df['ram_memory'] == selected_ram) & (df['gpu_brand'] == selected_gpu) &
                 (df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]

# Show filtered laptops
st.subheader("Matching Laptops")
st.dataframe(filtered_df[['brand', 'Model', 'Price', 'processor_brand', 'ram_memory', 'gpu_brand']])

# Laptop selection for recommendation
if not filtered_df.empty:
    selected_laptop = st.selectbox("Select a Laptop for Recommendations", filtered_df['Model'].unique())
    if st.button("Get Recommendations"):
        selected_index = df[df['Model'] == selected_laptop].index[0]
        recommendations = get_recommendations(selected_index)
        st.subheader("Recommended Laptops")
        st.dataframe(recommendations[['brand', 'Model', 'Price', 'processor_brand', 'ram_memory', 'gpu_brand']])
