import streamlit as st

# Add dynamic background color change with grey-to-blue gradient and subtle animation
from streamlit_lottie import st_lottie
import json

# Define the gradient background and top bar with the updated title
from PIL import Image
import streamlit as st
from PIL import Image
import base64

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
    #layout="wide"
)



# Load the Lottie animation
with open("lottie/Data_AI.json", "r") as f:
    emr_animation = json.load(f)


# Load the Lottie animation
with open("lottie/AI_EMR_PINK_BLUE.json", "r") as f:
    emr_pink_blue = json.load(f)

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

    # Load the logo image (if needed)
import streamlit as st
from streamlit_lottie import st_lottie
import json



# Load the Lottie animation for the welcome page
with open("lottie/Data_AI.json", "r") as f:
    welcome_animation = json.load(f)

# Custom CSS for the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #c0abcc;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content h2 {
        font-family: 'Roboto', sans-serif;
        font-size: 24px;
        color: #c0abcc;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content .stButton>button {
        background-color: #c0abcc;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .sidebar .sidebar-content .stButton>button:hover {
        background-color: #0FFCBE;
        color: #FFFFFF;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content .st-alert {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        color: #333333;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)







# Main content

logo_path = 'logos/logo-color.png'
logo_image = Image.open(logo_path)
st.image(logo_image, use_column_width=True)
st_lottie(welcome_animation, height=300, key="welcome-animation")



st.markdown("""
### InsightCare.AI is your personalized AI-powered decision support system.
Navigate through the app using the sidebar to explore various features.
""")


st.sidebar.success("Select a demo above.")
# Sidebar with Lottie animation



