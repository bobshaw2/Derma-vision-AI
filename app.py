
import os
import io
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model import load_model, predict_image


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="DermaVision AI | Skin Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "skin_model.pth"
ASSETS_DIR = BASE_DIR / "assets"


# -----------------------------------
# USER-FRIENDLY LESION NAMES
# -----------------------------------

FRIENDLY_NAMES = {
    "Actinic keratoses": "Sun-Damaged Skin Lesion",
    "Basal cell carcinoma": "Basal Cell Skin Cancer",
    "Benign keratosis-like lesions": "Non-Cancerous Skin Growth",
    "Dermatofibroma": "Usually Harmless Skin Bump",
    "Melanoma": "Melanoma (Serious Skin Cancer)",
    "Melanocytic nevi": "Common Mole (Pigmented Skin Spot)",
    "Vascular lesions": "Blood Vessel-Related Skin Mark",
}


def friendly_name(name):
    return FRIENDLY_NAMES.get(name, name)


# -----------------------------------
# IMAGE HELPERS
# -----------------------------------

def get_image_data(filename):
    """
    Convert a local image into a data URL
    that can be used inside custom HTML/CSS.
    """

    path = ASSETS_DIR / filename

    if not path.is_file():
        return None

    extension = path.suffix.lower()

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    mime = mime_types.get(extension)

    if mime is None:
        return None

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def feature_image(filename, fallback_icon):
    """
    Show an image beneath a feature card.
    Display a placeholder if the file is missing.
    """

    source = get_image_data(filename)

    if source:
        st.markdown(
            f"""
            <div
                class="feature-image"
                style="background-image:url('{source}');"
            ></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="feature-placeholder">
                {fallback_icon}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------
# PREMIUM CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            #ede9fe 0%,
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #faf9ff 0%,
            #f5f3ff 55%,
            #ffffff 100%
        );
}

/* WHITE SIDEBAR — ORIGINAL TEMPLATE */

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e9e5f5;
}

[data-testid="stSidebar"] * {
    color: #33265c;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

h1, h2, h3 {
    color: #241c47 !important;
    letter-spacing: -0.7px;
}

h1 {
    font-weight: 800 !important;
}

p {
    color: #5b5870;
}

/* ORIGINAL PURPLE GRADIENT HERO */

.hero {
    background: linear-gradient(
        120deg,
        #6d5dfc,
        #9276fa,
        #b69cff
    );
    padding: 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow:
        0 15px 35px rgba(109,93,252,0.17);
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #f4f0ff;
    font-size: 16px;
}

/* HERO WITH IMAGE */

.hero-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 25px;
}

.hero-text {
    flex: 1;
    min-width: 0;
}

.hero-visual {
    width: 39%;
    min-width: 220px;
    height: 225px;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 18px;
    background-color: #ded7ff;
    border: 2px solid rgba(255,255,255,0.45);
    box-shadow:
        0 10px 25px rgba(35,20,100,0.15);
}

.hero-placeholder {
    width: 39%;
    min-width: 220px;
    height: 225px;
    border-radius: 18px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 65px;
    background: rgba(255,255,255,0.18);
}

/* ORIGINAL GLASS CARDS */

.glass-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid #ebe7fa;
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
    box-shadow:
        0 8px 28px rgba(45,35,100,0.05);
}

.card-label {
    font-size: 13px;
    color: #817c99;
    font-weight: 600;
}

.card-value {
    font-size: 26px;
    font-weight: 800;
    color: #352866;
    margin-top: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #241c47;
    margin-top: 24px;
    margin-bottom: 15px;
}

/* FEATURE IMAGES */

.feature-image {
    height: 175px;
    width: 100%;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 18px;
    margin-top: 2px;
    border: 1px solid #ebe7fa;
    box-shadow:
        0 8px 20px rgba(45,35,100,0.06);
}

.feature-placeholder {
    height: 175px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 52px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #ede9fe,
        #dbeafe
    );
}

/* MODEL STATUS */

.status-online {
    display: inline-block;
    padding: 8px 14px;
    background: #e5f8ef;
    color: #08784d;
    border-radius: 30px;
    font-weight: 600;
}

.status-offline {
    display: inline-block;
    padding: 8px 14px;
    background: #fff0eb;
    color: #aa482c;
    border-radius: 30px;
    font-weight: 600;
}

/* BUTTONS */

div.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #6d5dfc,
        #9575fa
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 22px;
    font-weight: 700;
    box-shadow:
        0 5px 16px rgba(109,93,252,0.18);
}

div.stButton > button[kind="primary"]:hover {
    color: white;
    background: #5947e9;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 18px;
    padding: 14px;
    border: 1px solid #e7e0fb;
}

/* METRICS */

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #ebe7fa;
    border-radius: 16px;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #9690a9;
    padding: 25px;
    font-size: 12px;
}

/* MOBILE RESPONSIVENESS */

@media (max-width: 850px) {

    .hero-content {
        flex-direction: column;
        align-items: stretch;
    }

    .hero-visual,
    .hero-placeholder {
        width: 100%;
        min-width: 0;
    }

    .hero h1 {
        font-size: 29px;
    }

    .hero {
        padding: 24px;
    }
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# MODEL
# -----------------------------------

@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


def model_exists():
    return MODEL_PATH.is_file()


# -----------------------------------
# SESSION HISTORY
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.markdown("## 🔬 DermaVision")

    st.caption("AI SKIN INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Skin Analysis",
            "Analysis History",
            "About AI",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("AI MODEL STATUS")

    if model_exists():

        st.markdown(
            '<span class="status-online">'
            '● Checkpoint Found'
            '</span>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<span class="status-offline">'
            '● Model Not Connected'
            '</span>',
            unsafe_allow_html=True,
        )

    st.caption("EfficientNet-B0 | HAM10000")

    st.caption(
        "Research prototype • Not for diagnosis"
    )


# -----------------------------------
# DASHBOARD
# -----------------------------------

if page == "Dashboard":


    hero_source = get_image_data("hero.png")

    if hero_source:
        hero_visual = (
            f'<div class="hero-visual" '
            f'style="background-image:url({hero_source});">'
            f'</div>'
        )
    else:
        hero_visual = (
            '<div class="hero-placeholder">🔬 🧬</div>'
        )

    hero_html = f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-text">
            <h1>Welcome to DermaVision AI</h1>
            <p>
                Intelligent skin lesion image analysis
                powered by deep learning.
            </p>
            <p>
                Explore AI-assisted image classification
                through a modern research platform.
            </p>
        </div>
        {hero_visual}
    </div>
</div>
"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )


    # PLATFORM OVERVIEW

    st.markdown(
        '<div class="section-title">'
        'Platform Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-label">
                    AI Architecture
                </div>
                <div class="card-value">
                    EfficientNet-B0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-label">
                    Supported Categories
                </div>
                <div class="card-value">
                    7 Classes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        count = len(st.session_state.history)

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">
                    Session Analyses
                </div>
                <div class="card-value">
                    {count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # FEATURE CARDS

    st.markdown(
        '<div class="section-title">'
        'Explore Our Features'
        '</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="glass-card">
                <h3>🔬 AI Analysis</h3>
                <p>
                    Analyze uploaded lesion images
                    using a trained convolutional
                    neural network.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_image(
            "analysis.png",
            "🔬",
        )

    with b:

        st.markdown(
            """
            <div class="glass-card">
                <h3>📊 Confidence Insights</h3>
                <p>
                    Explore the model's predicted
                    class probability distribution.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_image(
            "technology.png",
            "📊",
        )

    with c:

        st.markdown(
            """
            <div class="glass-card">
                <h3>📁 Analysis History</h3>
                <p>
                    Review analysis results from
                    your current application session.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feature_image(
            "dermatologist.png",
            "🩺",
        )

    st.info(
        "Use the Skin Analysis page to upload "
        "an image and run the trained CNN."
    )


# -----------------------------------
# SKIN ANALYSIS
# -----------------------------------

elif page == "Skin Analysis":

    st.title("Skin Image Analysis")

    st.write(
        "Upload a clear skin lesion image "
        "to explore AI-generated classification results."
    )

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:

        st.markdown("### Upload Image")

        uploaded_file = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
        )

        image = None

        if uploaded_file is not None:

            if uploaded_file.size > 10 * 1024 * 1024:

                st.error(
                    "Maximum image size is 10 MB."
                )

            else:

                try:

                    image = Image.open(
                        io.BytesIO(
                            uploaded_file.getvalue()
                        )
                    ).convert("RGB")

                    st.image(
                        image,
                        caption="Image Preview",
                        use_container_width=True,
                    )

                except (
                    UnidentifiedImageError,
                    OSError,
                ):

                    st.error(
                        "Invalid image file."
                    )

        analyze = st.button(
            "✨ Analyze Skin Image",
            type="primary",
            use_container_width=True,
            disabled=image is None,
        )

    with right:

        st.markdown("### AI Analysis")

        if not analyze:

            st.info(
                "Your analysis results will "
                "appear here after processing."
            )

        elif not model_exists():

            st.error(
                "Trained model not found. "
                "Please save skin_model.pth "
                "inside the models folder."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing image..."
                ):

                    model, device = get_model()

                    result = predict_image(
                        image,
                        model,
                        device,
                    )

                original_prediction = result[
                    "prediction"
                ]

                prediction = friendly_name(
                    original_prediction
                )

                confidence = (
                    result["confidence"] * 100
                )

                st.success(
                    "Analysis completed"
                )

                st.metric(
                    "AI-Suggested Lesion Category",
                    prediction,
                )

                if prediction != original_prediction:

                    st.caption(
                        "Medical term: "
                        + original_prediction
                    )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                )

                st.progress(
                    min(
                        max(
                            result["confidence"],
                            0,
                        ),
                        1,
                    )
                )

                st.caption(
                    "Confidence is a model score, "
                    "not the probability of a "
                    "correct medical diagnosis."
                )

                probabilities = pd.DataFrame(
                    {
                        "Category": [
                            friendly_name(name)
                            for name in result[
                                "probabilities"
                            ].keys()
                        ],
                        "Probability (%)": [
                            value * 100
                            for value in result[
                                "probabilities"
                            ].values()
                        ],
                    }
                )

                st.markdown(
                    "### Class Probabilities"
                )

                st.bar_chart(
                    probabilities.set_index(
                        "Category"
                    )
                )

                st.session_state.history.append(
                    {
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "Prediction": prediction,
                        "Confidence (%)": round(
                            confidence,
                            2,
                        ),
                    }
                )

            except Exception as error:

                st.error(
                    "Model inference failed."
                )

                st.exception(error)

    st.divider()

    st.warning(
        "This AI system is for educational "
        "research only. It cannot diagnose "
        "or rule out skin cancer. Consult a "
        "qualified dermatologist for assessment, "
        "especially for changing, bleeding, "
        "or growing lesions."
    )


# -----------------------------------
# ANALYSIS HISTORY
# -----------------------------------

elif page == "Analysis History":

    st.title("Analysis History")

    st.caption(
        "Results are stored only during "
        "the current Streamlit session."
    )

    if not st.session_state.history:

        st.info(
            "No analyses recorded yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download History CSV",
            data=csv_data,
            file_name="dermavision_history.csv",
            mime="text/csv",
        )

        if st.button(
            "Clear Session History"
        ):

            st.session_state.history = []

            st.rerun()


# -----------------------------------
# ABOUT AI
# -----------------------------------

elif page == "About AI":

    st.title("About DermaVision AI")

    st.write(
        """
        DermaVision AI is an educational deep
        learning project that uses EfficientNet-B0
        to classify skin lesion images.
        """
    )

    st.markdown(
        "### Technology Stack"
    )

    st.write(
        "**Programming Language:** Python"
    )

    st.write(
        "**Frontend:** Streamlit"
    )

    st.write(
        "**Deep Learning:** PyTorch"
    )

    st.write(
        "**Architecture:** EfficientNet-B0"
    )

    st.write(
        "**Dataset:** HAM10000"
    )

    st.write(
        "**Classification:** 7 lesion categories"
    )

    st.markdown(
        "### AI Processing Pipeline"
    )

    st.code(
        """
Upload Skin Image
        |
Image Preprocessing
        |
EfficientNet-B0
        |
CNN Classification
        |
Class Probabilities
        |
Results Dashboard
        """,
        language="text",
    )

    st.markdown(
        "### Important Limitations"
    )

    st.write(
        """
        HAM10000 is a research dataset of skin
        lesion images. The model does not classify
        acne, eczema, or fungal infections.

        Model predictions can be incorrect and
        may perform differently across populations,
        skin tones, cameras, and image conditions.

        Clinical validation is required before
        any medical deployment.
        """
    )


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown(
    """
    <div class="footer">
        DermaVision AI © 2026<br>
        Powered by EfficientNet and PyTorch<br>
        Educational Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
