import streamlit as st
import google.generativeai as genai
import json
import re

# ========== Configuration ==========
genai.configure(api_key="AIzaSyCc4B5Og2hOxnERFBSp95iQ9aT-urSCKM8")
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ========== Text Cleaning ==========
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== Extract JSON ==========
def extract_json(response_text):
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            st.error("⚠️ No valid JSON found in Gemini's response.")
            return None
    except json.JSONDecodeError as e:
        st.error(f"⚠️ JSON Parsing Error: {str(e)}")
        return None

# ========== Gemini Classification ==========
def classify_complaint(complaint: str, user_type: str):
    prompt = f"""
    Classify the following food complaint:

    Complaint: "{complaint}"
    User Type: {user_type}

    Return the response as valid JSON **only**, following this structure:
    {{
        "department": "Predicted Department",
        "category": "User-Specific Category",
        "priority": "Priority Level (HIGH, MEDIUM, LOW, URGENT)",
        "keywords": ["List of related keywords"],
        "emergency_flag": true/false
    }}
    """
    try:
        response = model.generate_content(prompt)
        if not response or not response.text.strip():
            st.error("⚠️ Empty response from Gemini API.")
            return None

        raw_text = response.text.strip()

        with st.expander("🔍 Raw Gemini Response"):
            st.code(raw_text, language="json")

        return extract_json(raw_text)

    except Exception as e:
        st.error(f"❌ Error communicating with Gemini API: {str(e)}")
        return None

# ========== UI ==========

st.set_page_config(page_title="🍽️ Food Complaint Classifier", page_icon="🧠", layout="wide")

# ========== Header Section ==========
st.markdown("""
    <div style='text-align: center; padding: 1rem 0'>
        <h1 style='font-size: 2.8rem;'>🍽️ Food Complaint Classifier</h1>
        <p style='font-size: 1.2rem; color: #6c757d;'>Empowering smarter complaint handling using Gemini AI 🚀</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# ========== Input Section ==========
with st.container():
    st.subheader("📝 Enter Complaint Details")
    col1, col2 = st.columns([3, 1])
    with col1:
        complaint = st.text_area(
            "What's the issue?",
            placeholder="e.g., The packaging was damaged and the food was spoiled...",
            height=180
        )
    with col2:
        user_type = st.selectbox("👤 User Type", ["Consumer", "Retailer", "Supplier"])

# ========== Button + Result ==========
st.markdown("---")
if st.button("🔎 Analyze Complaint", use_container_width=True):
    if complaint.strip():
        cleaned = preprocess_text(complaint)
        with st.spinner("🤖 Thinking with Gemini..."):
            result = classify_complaint(cleaned, user_type)

        if result:
            st.success("✅ Complaint successfully classified!")
            
            with st.container():
                st.markdown("### 📊 Classification Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**🏢 Department:** `{result.get('department', 'N/A')}`")
                    st.markdown(f"**📌 Category:** `{result.get('category', 'N/A')}`")
                    st.markdown(f"**⚡ Priority:** `{result.get('priority', 'N/A')}`")

                with col2:
                    keywords = ", ".join(result.get("keywords", []))
                    st.markdown(f"**🔑 Keywords:** `{keywords}`")
                    emergency_flag = result.get("emergency_flag", False)
                    emoji_flag = "✅ Yes" if emergency_flag else "❌ No"
                    st.markdown(f"**🚨 Emergency:** {emoji_flag}")
        else:
            st.error("❌ Classification failed. Please try again.")
    else:
        st.warning("⚠️ Please enter a valid complaint.")

# ========== Footer ==========
st.markdown("""
    <hr>
    <div style='text-align: center; color: #888888; font-size: 0.9rem; padding-top: 10px'>
        Built with ❤️ using Streamlit & Gemini | v1.0
    </div>
""", unsafe_allow_html=True)
