import streamlit as st
import pandas as pd
import matplotlib as mp
def app(data):
    
    df = pd.DataFrame(data)
    cols = df.columns

    # table = st.table(df)
    st.write(df)

    print(cols)