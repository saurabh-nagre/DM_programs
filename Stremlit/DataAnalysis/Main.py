from json import load
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from multiapp import MultiApp
from Apps import asg6, asg7,asg8,none


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
    df = pd.read_csv(file)
    # st.header("Dataset Table")
    # st.dataframe(df, width=1000, height=500)
    return df


if file:
    data = load_file()
    
    app = MultiApp()

    app.add_app("Select Assignment",none.app)

    app.add_app("ASG6. AGNES, DIANA, K-MEAN,K-MEDOID, DBSCAN", asg6.app)
    app.add_app("ASG7. Apriori algorithm, Rules Generation", asg7.app)
    app.add_app("ASG8. BFS,DFS,Rank of Web Page,HITS Algorithm",asg8.app)
    # The main app
    app.run(data)
