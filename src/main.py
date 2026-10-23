import streamlit as st

def HomePage():
    st.title("Home page")



pg = st.navigation([
    st.Page(HomePage,title="Home Page"),
    st.Page("newFaceDetectWithNameDisplay.py",title="Streamlit WebRTC Persistent Face Tracker"),
    st.Page("registration.py",title="登録画面")
    
])
pg.run()