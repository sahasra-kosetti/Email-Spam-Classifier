import streamlit as st
import base64
import pickle
import os
import numpy as np
from io import StringIO

# Load the trained model and vectorizer
model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Function to add background image
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{encoded});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"⚠️ Background image '{image_file}' not found. Please check the file path.")

# Function to predict spam
def predict_spam(text):
    vectorized_text = cv.transform([text]).toarray()
    if hasattr(model, 'predict_proba'):
        spam_probability = model.predict_proba(vectorized_text)[0][1]  # Spam probability
    else:
        prediction = model.predict(vectorized_text)[0]
        spam_probability = 1.0 if prediction == 1 else 0.0  # Binary classification fallback
    
    # Ensure spam detection for common spam indicators
    spam_keywords = ["money", "win", "prize", "lottery", "cash", "100000", "free", "urgent", "congratulations"]
    if any(keyword in text.lower() for keyword in spam_keywords):
        spam_probability = max(spam_probability, 0.9)
    
    return spam_probability

# Main function
def main():
    add_bg_from_local("bg.jpg")

    st.sidebar.title("🎨 Theme Settings")
    theme = st.sidebar.radio("Choose a theme:", ["Light", "Dark"])

    if theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #1e1e1e; color: white; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.title("📧 Email Spam Classifier")
    st.subheader("🔍 Analyze your emails and detect spam instantly!")

    tab1, tab2 = st.tabs(["✏️ Type Email", "📂 Upload File"])

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    spam_score = None

    with tab1:
        msg = st.text_area("✉️ Enter email text:", st.session_state.input_text)
        if st.button("🔍 Predict"):
            if msg.strip() == "":
                st.warning("⚠️ Please enter text before predicting.")
            else:
                st.session_state.input_text = msg
                spam_score = predict_spam(msg)
                if spam_score > 0.5:
                    st.error(f"🚨 **SPAM DETECTED!** | Spam Probability: {spam_score:.2%}")
                else:
                    st.success(f"✅ **SAFE EMAIL** | Spam Probability: {spam_score:.2%}")

    with tab2:
        uploaded_file = st.file_uploader("📂 Upload a .txt file", type="txt")
        if uploaded_file is not None:
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.text_area("📄 File Content:", file_content, height=150)
            if st.button("📊 Analyze Uploaded Email"):
                spam_score = predict_spam(file_content)
                if spam_score > 0.5:
                    st.error(f"🚨 **SPAM DETECTED!** | Spam Probability: {spam_score:.2%}")
                else:
                    st.success(f"✅ **SAFE EMAIL** | Spam Probability: {spam_score:.2%}")

    if spam_score is not None:
        report = f"""
        📧 **Email Spam Classification Report**
        -------------------------------------
        Email Content:
        {msg if msg else file_content}

        🔍 Prediction:
        {'🚨 SPAM' if spam_score > 0.5 else '✅ HAM (Safe Email)'}

        📊 Spam Probability: {spam_score:.2%}
        """
        st.download_button("📥 Download Report", report, "spam_report.txt")

    if st.button("🧹 Clear"):
        st.session_state.input_text = ""
        st.rerun()

if __name__ == "__main__":
    main()                      