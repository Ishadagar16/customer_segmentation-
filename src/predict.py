import pickle
import pandas as pd


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

with open("model.pkl", "rb") as file:
    saved_model = pickle.load(file)

kmeans = saved_model["model"]
scaler = saved_model["scaler"]
features = saved_model["features"]


# --------------------------------------------------
# Customer Segment Names
# --------------------------------------------------

segment_names = {
    0: "Budget Customers",
    1: "Premium Customers",
    2: "Regular Customers",
    3: "Potential Customers"
}


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_customer_segment(
    age,
    income,
    recency,
    total_spending,
    total_purchases,
    deals_purchases,
    web_visits
):

    # Create input DataFrame
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

    # Scale input using the same scaler
    # used during model training
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Convert cluster number into segment name
    segment = segment_names.get(
        cluster,
        f"Customer Cluster {cluster}"
    )

    return cluster, segment


# --------------------------------------------------
# Main Program
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("CUSTOMER SEGMENTATION SYSTEM")
    print("=" * 50)

    print("\nEnter customer details:\n")

    age = float(input("Age: "))
    income = float(input("Annual Income: "))
    recency = float(input("Recency (Days Since Last Purchase): "))
    total_spending = float(input("Total Spending: "))
    total_purchases = float(input("Total Purchases: "))
    deals_purchases = float(input("Number of Deal Purchases: "))
    web_visits = float(input("Number of Web Visits per Month: "))

    cluster, segment = predict_customer_segment(
        age,
        income,
        recency,
        total_spending,
        total_purchases,
        deals_purchases,
        web_visits
    )

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print("Cluster Number:", cluster)
    print("Customer Segment:", segment)

    print("=" * 50)