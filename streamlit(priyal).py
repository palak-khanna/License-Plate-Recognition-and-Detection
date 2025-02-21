import streamlit as st
import pandas as pd


# Mock Data for Logs and Alerts
logs_data = pd.DataFrame({
    "License Plate": ["ABC123", "XYZ987", "LMN456"],
    "Entry Time": ["2025-01-26 10:00:00", "2025-01-26 10:30:00", "2025-01-26 11:00:00"],
    "Exit Time": ["2025-01-26 11:00:00", "", ""],
    "Duration": ["1 hour", "", ""],
    "Status": ["Exited", "Entered", "Entered"],
    "Alert": ["No", "No", "Prolonged Stay"]
})

alerts_data = pd.DataFrame({
    "License Plate": ["LMN456"],
    "Alert Type": ["Prolonged Stay"],
    "Detected Time": ["2025-01-26 12:00:00"],
    "Status": ["Active"]
})

# Header
st.title("Automatic License Plate Recognition System")

# Navigation Menu
tabs = ["Dashboard", "Logs", "Alerts", "Settings"]
selected_tab = st.sidebar.radio("Navigation", tabs)

# Dashboard Tab
if selected_tab == "Dashboard":
    st.header("Live Monitoring")
    
    # Live Camera Feed Simulation
    st.subheader("Camera Feed")
    st.image("https://via.placeholder.com/800x400?text=Live+Camera+Feed", caption="Real-Time Camera Feed")

    # Detected License Plate
    st.subheader("Detected License Plate")
    st.write("Currently Detected: `LMN456`")

    # Real-Time Logs
    st.subheader("Entry/Exit Logs")
    st.dataframe(logs_data)

# Logs Tab
elif selected_tab == "Logs":
    st.header("Vehicle Logs")

    # Search and Filter Options
    st.subheader("Search Logs")
    search = st.text_input("Search by License Plate or Date")

    # Filter Logs
    st.subheader("Logs Table")
    st.dataframe(logs_data)

    # Export Logs
    st.download_button("Download Logs", logs_data.to_csv(index=False), file_name="logs.csv", mime="text/csv")

# Alerts Tab
elif selected_tab == "Alerts":
    st.header("Active Alerts")

    # Display Active Alerts
    st.subheader("Ongoing Alerts")
    st.dataframe(alerts_data)

    # Acknowledge Alerts
    if st.button("Acknowledge All Alerts"):
        st.success("All alerts acknowledged.")

# Settings Tab
elif selected_tab == "Settings":
    st.header("System Settings")

    # Thresholds
    st.subheader("Alert Settings")
    prolonged_stay_threshold = st.slider("Prolonged Stay Threshold (hours)", 1, 12, 2)
    st.write(f"Current threshold: {prolonged_stay_threshold} hours")

    # Camera Management
    st.subheader("Camera Management")
    st.write("Manage connected cameras here.")
    camera_list = ["Camera 1", "Camera 2", "Camera 3"]
    selected_camera = st.selectbox("Select Camera", camera_list)

    # ML Models
    st.subheader("ML Model Management")
    st.file_uploader("Upload a new ML Model")
    st.button("Update Model")

