import streamlit as st
import pandas as pd
import os

# Import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Ml stuff
from pycaret.regression import setup, compare_models, pull, save_model
# from pycaret.classification import setup, compare_models, pull, save_model

# for animations

import json
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_le8PpGpm9v.json")

# #######################################################################################################

with st.sidebar:

    with st.container():
        st_lottie(lottie_hello)
    # st.image

    st.title("Auto ML Operations")
    choice = st.radio("Select Step wise : ", ["Upload", "Profiling", "ML","Download","About"])
    st.info("This application allows you to build an autometed ML pipeline \
    using Streamlit, Pandas Profiling and PyCaret.")

st.title('Auto Machine Learning Project')
# st.write("Auto Machine Learning Project")
st.caption('-By Saksham 732 (Msc DSAI Part-1) 2023')


with st.expander("See Steps (Important) : "):
    st.info("1. Select your data file (size upto 200mb).")
    st.info("2. You can see Data file preview down.")
    st.info("3. Next Click on , Profiling tab (on sidebar) to perform Autometed Exploratory Data Analysis.")
    st.info("4. To build ML model , click on ML tab and select your Target Variable.")
    st.info("5. Click on download to Download your trained Model in .pkl format.")


# ---------------------------------------------------------------------------------------------------------------

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.subheader("Upload Your Data for Modelling !")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col= None)
        df.to_csv("sourcedata.csv", index= None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Autometed Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target", df.columns)

    if st.button("Train Model"):
        setup(df, target=target)
        # setup(df, target=target, silent=True)

        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model

        # to save model
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")


if choice == "About":
    st.subheader("About AutoML Application")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.warning("Profiling")

        st.info("1. Profiling helps you to perform Autometed Exploratory Data Analysis.")
        st.info("2. You can get complete overview of your data file such as variables , observations , missings etc etc.")
        st.info("3. We have Interactive visualizations , Correlations , Heatmaps and so on.")

    with col2:
        st.warning("ML")
        st.success("1. In ML tab It allows you to selct your target column to Build Training model.")
        st.success("2. After Building model it suggests best fit model for our data.")


    with col3:
        st.warning("Download")
        st.info("1. In Download section we have facility to download trainned model in .pkl file format.")
        st.info("2. Once we have Model built , we can use it for Predictions.")


    