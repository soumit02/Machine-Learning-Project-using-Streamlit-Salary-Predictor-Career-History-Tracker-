import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import pickle as pk
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# File paths
ORIGINAL_DATA = "Salary_Data.csv"
CONTRIB_DATA = "contributed_data.csv"
MODEL_FILE = "salary_project"
CONTRIB_MODEL_FILE = "contrib_model.pkl"

# Load original dataset and model
df = pd.read_csv(ORIGINAL_DATA)
reg2 = pk.load(open(MODEL_FILE, "rb"))

# Load contributed data or initialize
if os.path.exists(CONTRIB_DATA):
    contrib_df = pd.read_csv(CONTRIB_DATA)
else:
    contrib_df = pd.DataFrame(columns=["Name", "Experience", "Salary"])
    contrib_df.to_csv(CONTRIB_DATA, index=False)

# Train model on contributed data if exists
def train_contrib_model():
    if not contrib_df.empty:
        X = contrib_df[["Experience"]]
        y = contrib_df["Salary"]
        model = LinearRegression()
        model.fit(X, y)
        with open(CONTRIB_MODEL_FILE, "wb") as f:
            pk.dump(model, f)

train_contrib_model()

# App settings
st.set_page_config(page_title="Salary Predictor & Career Tracker", layout="wide")
st.title("💼 Salary Predictor & Career History Tracker")

# Navigation
page = st.sidebar.radio("📌 Navigation", ["Home", "Prediction", "Contribute", "Profile", "Contributed Data"])

# Home Section
if page == "Home":
    st.image("sal.jpg", width=800)
    if st.checkbox("📊 Show Original Dataset"):
        st.dataframe(df)

    graph = st.selectbox("📈 Choose Graph Type", ["Non-Interactive", "Interactive"])
    val = st.slider("🎚️ Filter by Experience (Years)", 0, 20)
    df_filtered = df[df["YearsExperience"] >= val]

    if graph == "Non-Interactive":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_filtered["YearsExperience"], df_filtered["Salary"], color='teal')
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_ylim(0)
        st.pyplot(fig)
    else:
        chart = alt.Chart(df_filtered).mark_circle(size=80, color="orange").encode(
            x="YearsExperience", y="Salary"
        ).properties(title="Interactive Experience vs. Salary Chart")
        st.altair_chart(chart, use_container_width=True)

# Prediction Section
elif page == "Prediction":
    st.header("🔮 Know Your Predicted Salary")
    model_type = st.radio("📦 Choose Model", ["Original Model", "Contributed Model"])
    experience = st.number_input("📌 Enter Your Experience (in years)", 0.0, 50.0, step=0.5)
    input_data = np.array(experience).reshape(1, -1)

    if st.button("Predict"):
        if model_type == "Original Model":
            salary = reg2.predict(input_data)[0]
        else:
            if os.path.exists(CONTRIB_MODEL_FILE):
                contrib_model = pk.load(open(CONTRIB_MODEL_FILE, "rb"))
                salary = contrib_model.predict(input_data)[0]
            else:
                st.warning("⚠️ No contributed model found. Please add some data first.")
                salary = None
        if salary is not None:
            st.success(f"✅ Your predicted salary is BDT {salary:,.2f}")

# Contribute Section
elif page == "Contribute":
    st.header("📤 Contribute to Dataset")
    name = st.text_input("🧑 Your Name")
    ex = st.number_input("📈 Your Experience", 0.00, 20.00)
    sal = st.number_input("💰 Your Salary", 0.00, 800000.00, step=1000.00)
    if st.button("Submit"):
        new_data = pd.DataFrame({"Name": [name], "Experience": [ex], "Salary": [sal]})
        new_data.to_csv(CONTRIB_DATA, mode='a', header=False, index=False)
        st.success("✔️ Thank you for your contribution!")
        st.balloons()
        train_contrib_model()  # retrain on new data

# Contributed Data Viewer
elif page == "Contributed Data":
    st.header("📚 Contributed Dataset")
    contrib_df = pd.read_csv(CONTRIB_DATA)
    if not contrib_df.empty:
        st.dataframe(contrib_df)
    else:
        st.info("ℹ️ No contributions yet.")

# Profile Section
elif page == "Profile":
    st.header("👤 My Career Profile")
    profile_file = "profile_data.csv"

    if not os.path.exists(profile_file):
        pd.DataFrame(columns=["Name", "Year", "Company", "Salary", "Promotion"]).to_csv(profile_file, index=False)

    profile_df = pd.read_csv(profile_file)

    with st.form("profile_form"):
        st.subheader("📌 Add a Career Record")
        name = st.text_input("🧑 Your Name")
        year = st.number_input("📅 Year", 1980, 2100, step=1)
        company = st.text_input("🏢 Company Name")
        salary = st.number_input("💰 Salary", 0.00, 10000000.00, step=1000.00)
        promotion = st.text_input("🎖️ Promotion or Role Change (Optional)", placeholder="e.g., Promoted to Team Lead")
        submitted = st.form_submit_button("✅ Save to Profile")

        if submitted:
            new_row = pd.DataFrame({
                "Name": [name],
                "Year": [year],
                "Company": [company],
                "Salary": [salary],
                "Promotion": [promotion]
            })
            new_row.to_csv(profile_file, mode='a', header=False, index=False)
            st.success("✔️ Career data saved!")

    st.markdown("---")
    st.subheader("🔍 View Any Profile History")
    user_name = st.text_input("Enter a name to view profile history")

    if user_name:
        profile_df = pd.read_csv(profile_file)
        user_data = profile_df[profile_df["Name"].str.lower() == user_name.lower()]
        if not user_data.empty:
            user_data_sorted = user_data.sort_values("Year")
            st.table(user_data_sorted)
            chart = alt.Chart(user_data_sorted).mark_line(point=True).encode(
                x="Year:O", y="Salary", tooltip=["Company", "Promotion"]
            ).properties(title="📊 Salary Trend for " + user_name.title())
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("❌ No profile data found for this name.")
