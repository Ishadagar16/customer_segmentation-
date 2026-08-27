import streamlit as st
import pandas as pd
import pickle


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

with open("model.pkl", "rb") as file:
    saved_model = pickle.load(file)

kmeans = saved_model["model"]
scaler = saved_model["scaler"]
features = saved_model["features"]


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="",
    layout="centered"
)


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("Customer Segmentation System")
st.write(
    "Enter customer information to identify the "
    "customer segment using K-Means Clustering."
)


# --------------------------------------------------
# User Input
# --------------------------------------------------

st.subheader("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

recency = st.number_input(
    "Recency (Days Since Last Purchase)",
    min_value=0,
    max_value=100,
    value=30
)

total_spending = st.number_input(
    "Total Spending",
    min_value=0.0,
    value=500.0,
    step=50.0
)

total_purchases = st.number_input(
    "Total Purchases",
    min_value=0,
    value=10,
    step=1
)

deals_purchases = st.number_input(
    "Number of Deal Purchases",
    min_value=0,
    value=3,
    step=1
)

web_visits = st.number_input(
    "Number of Web Visits per Month",
    min_value=0,
    value=5,
    step=1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Customer Segment"):

    input_data = pd.DataFrame(
        [[
            age,
            income,
            recency,
            total_spending,
            total_purchases,
            deals_purchases,
            web_visits
        ]],
        columns=features
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Cluster-to-segment mapping
    segment_names = {
        0: "Budget Customers",
        1: "Premium Customers",
        2: "Regular Customers",
        3: "Potential Customers"
    }

    segment = segment_names.get(
        cluster,
        f"Customer Cluster {cluster}"
    )

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success("Customer segmentation completed!")

    st.subheader("Prediction Result")

    st.write("**Cluster Number:**", cluster)

    st.write("**Customer Segment:**", segment)

    # --------------------------------------------------
    # Basic Business Recommendation
    # --------------------------------------------------

    st.subheader("Business Recommendation")

    if segment == "Premium Customers":

        st.write(
            "These customers have strong purchasing behavior. "
            "Businesses can provide loyalty rewards, premium "
            "offers and personalized recommendations."
        )

    elif segment == "Budget Customers":

        st.write(
            "These customers are more price-sensitive. "
            "Discounts, coupons and affordable product offers "
            "can help increase their purchases."
        )

    elif segment == "Regular Customers":

        st.write(
            "These customers show regular purchasing behavior. "
            "Businesses can use personalized offers and loyalty "
            "programs to increase their spending."
        )

    elif segment == "Potential Customers":

        st.write(
            "These customers may have potential for increased "
            "purchases. Targeted promotions and personalized "
            "marketing can help improve engagement."
        )