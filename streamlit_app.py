import streamlit as st
from brain_tumor_model import Brain_Model

st.title("Brain Tumor Prediction Model")

st.file_uploader("Please upload photo from your brain CT image", ["png", "jpg", "jpeg"],accept_multiple_files=False)

st.image("agyikepek_3_osztaly\\kepek\\1732_3.png", caption="Tumorous Brain")
st.write(f"The brain is tumorous with {Brain_Model.accuracy*100}% certainty")