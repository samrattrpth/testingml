from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_house_price_model.pkl"
DATASET_PATH = BASE_DIR / "dataset.pkl"

# Page Configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction | Enterprise AI", 
    page_icon="", 
    layout="centered"
)

# Custom Modern Enterprise SaaS CSS Styling
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header Typography */
    h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 5px;
    }

    /* Cards / Containers */
    div[data-testid="stForm"] {
        background: linear-gradient(145deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 35px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    }

    /* Form Section Header */
    .form-section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
        letter-spacing: -0.01em;
        border-left: 3px solid #6366f1;
        padding-left: 10px;
    }

    /* Input Field Labels */
    label p {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
    }

    /* Inputs & Selectboxes */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        background-color: #1a2234 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput input:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        letter-spacing: 0.01em;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Team Banner */
    .team-banner {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #6366f1;
        padding: 20px 24px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 30px;
        font-size: 0.875rem;
        color: #9ca3af;
    }
    .team-title {
        color: #e5e7eb;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .team-member {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
        max-width: 450px;
    }
    .team-member:last-child {
        border-bottom: none;
    }
    .member-name {
        color: #f3f4f6;
        font-weight: 500;
    }
    .member-uid {
        color: #818cf8;
        font-family: monospace;
    }
    .team-section {
        margin-top: 12px;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    .team-section b {
        color: #e5e7eb;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    /* Result Box Card */
    .result-card {
        background: linear-gradient(145deg, #111827 0%, #172033 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Team Information Banner (Structured in Vertical Rows)
st.markdown("""
    <div class="team-banner">
        <div class="team-title">Project Created by:</div>
        <div class="team-member">
            <span class="member-name">Samrat Tripathi</span>
            <span class="member-uid">UID: 26BAI70082</span>
        </div>
        <div class="team-member">
            <span class="member-name">Chandan Sharma</span>
            <span class="member-uid">UID: 26BAI70060</span>
        </div>
        <div class="team-member">
            <span class="member-name">Anshu Yadav</span>
            <span class="member-uid">UID: 26BAI70062</span>
        </div>
        <div class="team-section">
            <b>Section:</b> 26AML-101B
        </div>
    </div>
""", unsafe_allow_html=True)

# App Hero Header
st.title("House Price Prediction")
st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Estimate residential property valuations using advanced random forest machine learning benchmarks.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run the final export cell in the Jupyter notebook and place random_forest_house_price_model.pkl next to app.py.")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

@st.cache_data
def load_locations():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.pkl not found. Run the final export cell in the notebook and place dataset.pkl next to app.py.")
    with DATASET_PATH.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, pd.DataFrame) or "location" not in data.columns:
        raise ValueError("dataset.pkl must be a DataFrame containing a 'location' column.")
    return sorted(data["location"].dropna().astype(str).unique().tolist())

try:
    model = load_model()
    locations = load_locations()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Input Form (SaaS Card Layout)
with st.form("prediction_form"):
    st.markdown('<div class="form-section-title">Property Parameters</div>', unsafe_allow_html=True)
    
    location = st.selectbox("Location", locations)
    total_sqft = st.number_input("Total Area (sq. ft.)", min_value=150.0, max_value=100000.0, value=1000.0, step=50.0)
    
    col1, col2 = st.columns(2)
    with col1:
        bath = st.number_input("Bathrooms", min_value=1, max_value=30, value=2, step=1)
    with col2:
        bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=50, value=2, step=1)
        
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Estimate Property Price", use_container_width=True)

if submitted:
    if total_sqft / bhk < 150:
        st.warning("Notice: The area is unusually small for this number of bedrooms; prediction accuracy may vary.")
    if bath > bhk + 2:
        st.warning("Notice: This bathroom-to-bedroom combination is uncommon in historical training records.")
        
    features = pd.DataFrame(
        [[location, float(total_sqft), float(bath), float(bhk)]],
        columns=["location", "total_sqft", "bath", "BHK"],
    )
    
    try:
        predicted_lakh = float(model.predict(features)[0])
        if not pd.notna(predicted_lakh) or predicted_lakh <= 0:
            st.error("The model returned an invalid valuation. Check the dataset configuration.")
        else:
            st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; font-weight: 600; margin-bottom: 4px;">Valuation Result</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">₹{predicted_lakh:,.2f} <span style="font-size: 1.2rem; font-weight: 500; color: #9ca3af;">lakh</span></div>
                    <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 6px;">Approximate Equivalent: ₹{predicted_lakh * 100000:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Prediction execution failed: {exc}")