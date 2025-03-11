import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to AI-Powered Diet Recommendation System! 👋")
st.markdown(
    """
    A diet recommendation web application using content-based approach with Scikit-Learn, FastAPI and Streamlit.
    Discover personalized meal suggestions tailored to your dietary needs. Input your age, weight, and preferences to receive customized recipes that promote healthier eating. Enjoy a user-friendly interface and explore a variety of nutritious options today!
    """
)
