import pandas as pd
import pickle
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://media.licdn.com/dms/image/v2/D4D22AQGhTriDEECLTQ/feedshare-shrink_800/feedshare-shrink_800/0/1730298782626?e=1733356800&v=beta&t=yikOeQh9zlLIAPT7CS3Z1iaV9xmcZaSDyKHI6Q2bFpI');
    background-size: cover;
}

[data-testid="stSidebar"] {
    background-image: url('https://images.fineartamerica.com/images-medium-large/yellow-flower-blue-background-matthias-hauser.jpg');
    background-size: cover;
    color: white;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


with open('mobile_recommender_model.pkl', 'rb') as model_file:
    cosine_sim = pickle.load(model_file)

data = pd.read_csv('mobiles_dataset.csv')

def get_recommendations(product_index, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]  # Exclude the product itself
    recommended_indices = [i[0] for i in sim_scores]
    return data.iloc[recommended_indices]
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Recommendation", "Price Predictor"])

# Main Page
if app_mode == "Home":
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 80px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Mobile Recommendation System!</h1>
            <h3 style="color: black;">"Find the best mobile for your needs"</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display mobile images in columns with background styling
    st.markdown(
         """
         <div style="display: flex; justify-content: space-around; 
                     background-color: rgba(0, 0, 0, 0); padding: 20px; border-radius: 10px; 
                     margin-top: 20px;">
         """, unsafe_allow_html=True
     )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image('Home pics/phone1.png')
    with col2:
        st.image('Home pics/phone2.png')
    with col3:
        st.image('Home pics/phone3.png')
    with col4:
        st.image('Home pics/phone4.png')
    with col5:
        st.image('Home pics/phone5.png')

    st.markdown("</div>", unsafe_allow_html=True)

elif app_mode == "About":
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 30px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">About</h1>
            <p style="color: black;">In today's fast-paced digital world,
            selecting the right mobile device can be overwhelming due to the plethora of options available.
            A mobile recommendation system powered by machine learning can significantly enhance user experience by providing tailored recommendations
            based on user preferences and device features. This system leverages content-based collaborative filtering, 
            which combines the strengths of content-based filtering and collaborative filtering techniques.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
         """
         <div style="display: flex; justify-content: space-around; 
                     background-color: rgba(0, 0, 0, 0); padding: 10px; border-radius: 10px; 
                     margin-top: 10px;">
         """, unsafe_allow_html=True
     )
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 30px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Objectives</h1>
            <h3 style="color: black;">The primary objectives of the mobile recommendation system include:</h>
            <p style="color: black;"></p>
            <p style="color: black;">1.Personalized Recommendations: Provide users with mobile device suggestions that align with their preferences and needs.</p>
            <p style="color: black;">2.User Engagement: Increase user interaction with the platform by offering relevant recommendations.</p>
            <p style="color: black;">3.Market Insights: Gather data on user preferences to analyze trends in mobile device choices.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
         """
         <div style="display: flex; justify-content: space-around; 
                     background-color: rgba(0, 0, 0, 0); padding: 10px; border-radius: 10px; 
                     margin-top: 10px;">
         """, unsafe_allow_html=True
     )
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 30px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Implementation Steps</h1>
            <h3 style="color: black;">1.Data Preprocessing:</h>
            <p style="color: black;">Clean and preprocess the dataset, handling missing values and normalizing numerical features.</p>
            <h3 style="color: black;">2.Model Training:</h>
            <p style="color: black;">Use machine learning algorithms (e.g., KNN for collaborative filtering) to build the recommendation model.</p>
            <p style="color: black;">Implement techniques like cosine similarity or Jaccard similarity to compute the similarity between users and items.</p>
            <h3 style="color: black;">3.User Interface Development:</h>
            <p style="color: black;">Create a user-friendly interface using web frameworks like Streamlit, allowing users to input their preferences and view recommendations.</p>
            <p style="color: black;">Integrate visualization tools to display device comparisons and features.</p>
            <h3 style="color: black;">4.Evaluation:</h>
            <p style="color: black;">Evaluate the recommendation system using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) to assess accuracy.</p>
            <p style="color: black;">Conduct A/B testing to compare the effectiveness of the recommendation engine against a baseline.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
         """
         <div style="display: flex; justify-content: space-around; 
                     background-color: rgba(0, 0, 0, 0); padding: 10px; border-radius: 10px; 
                     margin-top: 10px;">
         """, unsafe_allow_html=True
     )
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 30px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Made By</h1>
            <h3 style="color: black;">Name: Advitiya Arya</h3>
            <h3 style="color: black;">Subject: Statistical Machine Learning [CSET211]</h3>
            <h3 style="color: black;">Enrollment No.: E23CSEU1110</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

elif app_mode == "Recommendation":
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 20px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Mobile Recommender</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    <style>
    .selectbox-label {
        color: black; /* Set text color to black */
        font-weight: bold; /* Make text bold */
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="selectbox-label">Select a product:</p>', unsafe_allow_html=True)
    # Select a product to recommend from
    product_names = data['name'].tolist()
    selected_product = st.selectbox("", product_names)

    if st.button('Show Similar Phones'):
        product_index = data[data['name'] == selected_product].index[0]
        recommended_products = get_recommendations(product_index)

        st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 5px; 
            border-radius: 10px;
        ">
            <h4 style="color: black;">Recommended Products</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
         """
         <div style="display: flex; justify-content: space-around; 
                     background-color: rgba(0, 0, 0, 0); padding: 10px; border-radius: 10px; 
                     margin-top: 10px;">
         """, unsafe_allow_html=True
     )
        for index, row in recommended_products.iterrows():
            product_div = f"""
            <div style="
                background-color: rgba(125, 133, 129, 0.9); 
                border-radius: 20px;
                padding: 5px;
                margin: 20px;
            ">
                <h3 style="margin: 0;">{row['name']}</h3>
                <p><strong>Price:</strong> {row['price']}</p>
                <p><strong>Stars:</strong> {row['stars']}</p>
                <p><strong>Description:</strong> {row['desc']}</p>
                <p><a href="{row['url']}" style="text-decoration: none; color: blue;">View Product</a></p>
            </div>
            """
    
            # Display the product div in Streamlit
            st.markdown(product_div, unsafe_allow_html=True)
            st.image("Home pics/phone1.png", width=300)#Demo image
elif app_mode == "Price Predictor":
    st.markdown(
        """
        <div style="
            background-color: rgba(255, 255, 255, 0.9); 
            padding: 20px; 
            border-radius: 10px;
        ">
            <h1 style="color: black;">Mobile Price Predictor</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <style>
    .selectbox-label {
        color: black; /* Set text color to black */
        font-weight: bold; /* Make text bold */
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="selectbox-label">Select RAM (in GB)</p>', unsafe_allow_html=True)
    ram_options = [4, 8, 12, 24]
    ram = st.selectbox("", ram_options)

    st.markdown("""
    <style>
    .selectbox-label {
        color: black; /* Set text color to black */
        font-weight: bold; /* Make text bold */
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="selectbox-label">Select Storage (in GB)</p>', unsafe_allow_html=True)
    storage_options = [32, 64, 128, 256, 512, 1024]
    storage = st.selectbox("", storage_options)

    st.markdown("""
    <style>
    .selectbox-label {
        color: black; /* Set text color to black */
        font-weight: bold; /* Make text bold */
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="selectbox-label">Display Size (in inches)</p>', unsafe_allow_html=True)
    display = st.number_input("", min_value=0.0, max_value=10.0, value=6.0)

    if st.button("Predict Price"):
        
        input_data = pd.DataFrame([[ram, storage, display]], columns=['RAM', 'storage', 'display'])
        
       
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        predicted_price = model.predict(input_data_scaled)
        
        # Display the result
        st.markdown(f"""
        <div style='background-color: white; padding: 6px; border-radius: 10px;'>
            <h3 style='color: black;'>The predicted price of the mobile is: ₹{predicted_price[0]:.2f}</h3>
        </div>
    """, unsafe_allow_html=True)    