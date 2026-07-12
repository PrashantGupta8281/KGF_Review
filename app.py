import streamlit as st
import pandas as pd
from transformers import pipeline
import evaluate
from PIL import Image

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION & CUSTOM CINEMATIC THEME (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K.G.F. 2 Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection of CSS to enforce stunning dark theme, styled buttons & fonts
st.markdown("""
    <style>
        /* Base Page Background and Fonts */
        .stApp {
            background-color: #0d0d0d;
            color: #f1f1f1;
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        
        /* Headers and Typography */
        h1, h2, h3 {
            color: #ff9900 !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Modern Glassmorphic Container styling */
        div[data-testid="stVerticalBlock"] > div {
            background: rgba(25, 25, 25, 0.65);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 153, 0, 0.15);
            margin-bottom: 1rem;
        }
        
        /* Custom Styling for Streamlit Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #ff9900 0%, #b36b00 100%) !important;
            color: #ffffff !important;
            font-weight: bold !important;
            font-size: 16px !important;
            padding: 0.6rem 2.5rem !important;
            border-radius: 30px !important;
            border: none !important;
            box-shadow: 0px 4px 15px rgba(255, 153, 0, 0.3) !important;
            transition: all 0.3s ease-in-out !important;
            width: 100%;
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0px 6px 20px rgba(255, 153, 0, 0.5) !important;
            color: #000000 !important;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #141414 !important;
            border-right: 1px solid #262626;
        }
        
        /* Success/Danger styling alterations */
        .stAlert {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHING TRANSFORMERS PIPELINE & METRICS FOR PERFORMANCE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    # Utilizing your chosen DistilBERT Model fine-tuned for SST-2
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

@st.cache_resource
def load_evaluation_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    return accuracy, f1

classifier = load_sentiment_pipeline()
accuracy_metric, f1_metric = load_evaluation_metrics()

# -----------------------------------------------------------------------------
# 3. APP HEADER & THE MOVIE POSTER HERO BANNER
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 2.2])

with col1:
    try:
        # Looking for the local image file 
        poster_img = Image.open("kgf2_poster.png")
        st.image(poster_img, use_container_width=True, caption="K.G.F Chapter 2 - The Legacy Continues")
    except FileNotFoundError:
        # Fallback empty visual element placeholder if file isn't found
        st.warning("⚠️ For a fully stunning visual layout, place the K.G.F 2 poster image as 'kgf2_poster.jpg' in the app directory.")

with col2:
    st.title("🎬 K.G.F 2 Movie Review Analyzer")
    st.subheader("Powered by LLMs & Generative AI Pipelines")
    st.markdown("""
    Welcome to the ultimate NLP dashboard for analyzing **K.G.F. Chapter 2** movie reception. 
    This application transforms native text classification pipelines into real-time business insights. 
    
    **Demonstrated Advanced Tasks:**
    * 🧠 **Sentiment Analysis:** Real-time Positivity vs Negativity tracking.
    * 📊 **Pipeline Metrics Evaluation:** Live Ground-truth Accuracy & $F_1$ verification scores.
    * 📈 **Batch Inferencing:** Bulk processing capability for dynamic file sets.
    """)

st.write("---")

# -----------------------------------------------------------------------------
# 4. SIDEBAR SETTINGS & BULK DATASET INGESTION
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Operational Controls")

# Optional bulk file uploader mimicking your Pandas loading block
uploaded_file = st.sidebar.file_uploader("Upload review dataset (.csv)", type=["csv"])

# Instantiating default sample data frame to match notebook context smoothly
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Creating a sample mock dataset if no file is uploaded yet
    mock_data = {
        'Review': [
            "KGF 2 is an amazing movie with powerful action and excellent performance.",
            "Rocky Bhai is exceptional, the background score gave me literal goosebumps!",
            "Too loud and too long. The action sequences felt overly drawn out and repetitive.",
            "An absolute masterpiece of commercial Indian cinema. Visual spectacle at its peak.",
            "The plot was extremely weak and predictable. Disappointed considering the massive hype."
        ],
        'Class': ["POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]
    }
    df = pd.DataFrame(mock_data)

# -----------------------------------------------------------------------------
# 5. MAIN SYSTEM WORKFLOWS (TAB PANELS ACCORDING TO NLP TASKS)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔥 Live Playground Inference", "📊 Dataset Ingestion & Evaluation Metrics"])

with tab1:
    st.write("### 🧪 Real-time Sentiment Classifier Playground")
    st.write("Type or paste any customer review below to test the active *DistilBERT* Transformer instance pipeline.")
    
    # Custom interactive live input 
    user_review_input = st.text_area("User Review Input:", 
                                     value="KGF 2 is an amazing movie with powerful action and excellent performance.")
    
    # Run Inference button
    if st.button("🚀 Analyze Single Review Sentiment"):
        with st.spinner("Model inferencing in progress..."):
            prediction_output = classifier(user_review_input)[0]
            
            # Formulating output design components
            label = prediction_output['label']
            score = prediction_output['score']
            
            st.write("#### Result:")
            if label == "POSITIVE":
                st.success(f"🌟 **Sentiment Result:** {label} (Confidence Score: {score:.4f})")
            else:
                st.error(f"🚨 **Sentiment Result:** {label} (Confidence Score: {score:.4f})")

with tab2:
    st.write("### 📦 Bulk Dataset Processing & Architecture Evaluation")
    st.write("Below is the actively ingested movie review pipeline matrix data framing structure.")
    
    # Data View Container
    st.dataframe(df, use_container_width=True)
    
    if st.button("⚙️ Execute Model Pipeline Batch Run"):
        if 'Review' in df.columns and 'Class' in df.columns:
            with st.spinner("Processing batch datasets over Hugging Face pipeline..."):
                reviews_list = df['Review'].tolist()
                real_labels = df['Class'].tolist()
                
                # Perform Pipeline Batch Inference
                predicted_outputs = classifier(reviews_list)
                
                # Constructing structural mapping parameters matching original variables
                references = [1 if lbl == "POSITIVE" else 0 for lbl in real_labels]
                predictions = [1 if pred['label'] == "POSITIVE" else 0 for pred in predicted_outputs]
                
                # Computing accuracy & f1 parameters safely
                accuracy_res = accuracy_metric.compute(references=references, predictions=predictions)['accuracy']
                f1_res = f1_metric.compute(references=references, predictions=predictions)['f1']
                
                # Dynamic Metric Cards rendering
                st.write("#### 🎯 Execution Model Performance Metrics")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(label="📊 Pipeline Accuracy Score", value=f"{accuracy_res * 100:.2f} %")
                m_col2.metric(label="🧪 F1 Evaluation Score", value=f"{f1_res:.4f}")
                
                # Build annotated summary table output
                results_df = df.copy()
                results_df['Predicted Sentiment'] = [p['label'] for p in predicted_outputs]
                results_df['Confidence Confidence'] = [f"{p['score']:.4f}" for p in predicted_outputs]
                
                st.write("#### 📝 Pipeline Predictions Logs Matrix Output")
                st.dataframe(results_df, use_container_width=True)
        else:
            st.error("❌ Fatal Error: Uploaded dataset must contain precisely labeled columns titled 'Review' and 'Class'.")

# -----------------------------------------------------------------------------
# 6. FOOTER BRAND ELEMENT
# -----------------------------------------------------------------------------
st.markdown("""
    <br><hr>
    <div style='text-align: center; color: #555555; font-size: 12px;'>
        K.G.F 2 LLM Classifier Dashboard Core Engine • Crafted using Streamlit and Hugging Face Transformers.
    </div>
""", unsafe_allow_html=True)
