import os
import streamlit as st
import requests

# Load API base URI from secrets or environment
api_key = os.environ.get('API_URI', 'cloud_api_uri')
BASE_URI = st.secrets.get(api_key)
BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'

st.title("Medical Imaging AI App 🧠")
st.markdown(f"Backend: `{BASE_URI}`")

st.markdown("Upload a medical image and select a task (classification or segmentation).")

# Upload image
uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

# Task selection
task = st.selectbox("Select a task", ["Classification", "Segmentation"])

if uploaded_file and task:
    endpoint = "classification" if task == "Classification" else "segmentation"
    url = BASE_URI + endpoint

    st.markdown(f"**Sending request to:** `{url}`")

    with st.spinner("Processing..."):
        try:
            files = {'img': uploaded_file}
            response = requests.post(url, files=files)
        except Exception as e:
            st.error(f"❌ Failed to contact API: {e}")
            st.stop()

        if response.status_code != 200:
            st.error(f"❌ API error {response.status_code}: {response.text}")
            st.stop()

        # ✅ Classification result
        if task == "Classification":
            try:
                result = response.json()
                prob = result.get("Probability of Malignant Breast Cancer")

                if prob is None:
                    st.error("✅ API returned successfully, but no probability was found in the response.")
                    st.write("Full response:", result)
                else:
                    st.success(f"🎯 Prediction: **{prob * 100:.2f}% malignant**")
            except Exception as e:
                st.error(f"❌ Failed to parse prediction response: {e}")
                st.write("Raw response content:", response.text)

        # ✅ Segmentation result (image)
        elif task == "Segmentation":
            st.image(response.content, caption="🩺 Segmentation Output", use_column_width=True)
