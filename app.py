import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression


st.title("HDAG Discovery Week 3-4")


st.subheader("*I kinda know how to do StreamLit now!*")
st.write("")

df = pd.read_csv("Affordable_Housing_by_Town_2011-2022.csv")
df = df.dropna()
df.columns = df.columns.str.replace(r"\s+", "", regex=True)
df.columns = df.columns.str.replace(r"[^\w%]", "", regex=True)
st.write(df)
st.markdown("""
**See data source here:** [Kaggle](https://www.kaggle.com/datasets/utkarshx27/affordable-housing-by-town-2011-2022/data)
""")

st.title("I love school!")
school = st.slider("How much do you love failing MATH 22A", 0, 100, 50)
if school < 50:
    st.write("No! You love failing 22A!")
else:
    st.write("Good!")

st.title("Affordability% vs Selected Metric:")


x_var = st.selectbox(
    "Select X-axis variable:",
    [
        "DeedRestrictedUnits",
        "TenantRentalAssistance",
        "SingleFamilyCHFAUSDAMortgages",
        "GovernmentAssisted",
        "TotalAssistedUnits"
    ]
)

st.write(f"### Showing: **{x_var}** vs **Affordability%**")


x = df[x_var]
y = df["PercentAffordable"]


model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)
df["y_pred"] = model.predict(x.values.reshape(-1, 1))



scatter = alt.Chart(df).mark_circle(size=70).encode(
    x=alt.X(x_var, title=x_var),
    y=alt.Y("PercentAffordable", title="Affordability (%)"),
)


line = alt.Chart(df).mark_line(color="red", strokeWidth=3).encode(
    x=x_var,
    y="y_pred"
)

chart = (scatter + line).properties(width="container", height=450)

st.altair_chart(chart, use_container_width=True)