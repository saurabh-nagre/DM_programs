import streamlit as st
import pandas as pd
from multiapp import MultiApp
from Apps import SFdataAnalysis


# variables
disrad = False

st.title("Data Analysis Tool")

file = st.file_uploader("Enter Dataset first to Proceed", type=[
                        'csv','txt'], accept_multiple_files=False, disabled=False)
# data = pd.read_csv(file)



def load_file():
    # print(file.type)
    if file.type=="text/plain":
        return file
    df = pd.read_csv(file,low_memory=False)
    # st.header("Dataset Table")
    # st.dataframe(df, width=1000, height=500)
    return df


if file:
    data = load_file()
    
    app = MultiApp()

    app.add_app("San fransisco Salaries Data Analysis",SFdataAnalysis.app)
    
    # The main app
    app.run(data)
