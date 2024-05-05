import streamlit as st
from PIL import Image
import base64
from pathlib import Path
import os

@st.cache_data(show_spinner=False)
def render_sidebar():
    current_path = os.getcwd()
    logo_path = current_path + "/image/EC_challenge_logo.png"
    
    with open(logo_path, "rb") as f:
        logo = base64.b64encode(f.read()).decode("utf-8")
    
    sidebar_markdown = f"""
    
    <center>
    <img src="data:image/png;base64,{logo}" width="100" height="100" />
    <h1>
    About
    </h1>  
    &nbsp;    
    </center>
    
    <hr>
    
    <p>
    Hackathon challenge \n
    - Customer experience improvement with LLM
    </p> 
    
    <center>
    <a href="https://github.com/gary8564">
    <img src = "https://cdn-icons-png.flaticon.com/512/733/733609.png" width="23" /></a>
    
    <a href="mailto:chia.hao.chang@rwth-aachen.de">
    <img src="https://cdn-icons-png.flaticon.com/512/646/646094.png" alt="email" width = "27" ></a>
    </center>
    
    &nbsp;


    """

    st.sidebar.markdown(sidebar_markdown, unsafe_allow_html=True)
    
    