import streamlit as st
import requests
with st.spinner("Loading model, it might take a while..."):
    response = requests.get("http://127.0.0.1:9090/")
    st.success(response.json()["message"])

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("**Accuracy: 82.9%**")  
st.write("### Supported Categories:")
st.write("Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck")
st.info("⚠️ Note: Images outside these categories may result in inaccurate classification") 

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if st.button("Get prediction"):
    if uploaded_image is not None:
        url = "http://127.0.0.1:9090/predict"
        files = {"image":(uploaded_image.name, uploaded_image.read(), uploaded_image.type)}
        with st.spinner("Analyzing image...."):
            response = requests.post(url, files=files)
        if response.status_code == 200:
            st.success("✅ Prediction complete!")
            st.write(f"**Result:** {response.json()}") 
        else:
            st.write("❌ Request failed with status code", response.status_code)
            st.write(response.text)
    else:
        st.write("Please upload an image")