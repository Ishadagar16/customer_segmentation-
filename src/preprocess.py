import pandas as pd
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(file_path="customer_segmentation.csv"):
    """
    Load customer dataset.
    """

    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception:
        df = pd.read_csv(
            file_path,
            sep=None,
            engine="python"
        )

    return df


# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------

def preprocess_data(df):
    """
    Clean the customer dataset and create
    features required for clustering.
    """

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Handle missing Income values
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(
            df["Income"].median()
        )

    # Convert customer date
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(
            df["Dt_Customer"],
            format="%d-%m-%Y",
            errors="coerce"
        )

    # --------------------------------------------------
    # Create Age
    # --------------------------------------------------

    if "Year_Birth" in df.columns:
        current_year = 2015

        df["Age"] = (
            current_year - df["Year_Birth"]
        )

        # Remove unrealistic ages
        df = df[
            (df["Age"] >= 18) &
            (df["Age"] <= 100)
        ].copy()

    # --------------------------------------------------
    # Create Total Spending
    # --------------------------------------------------

    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    available_spending_columns = [
        column for column in spending_columns
        if column in df.columns
    ]

    df["Total_Spending"] = df[
        available_spending_columns
    ].sum(axis=1)

    # --------------------------------------------------
    # Create Total Purchases
    # --------------------------------------------------

    purchase_columns = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]

    available_purchase_columns = [
        column for column in purchase_columns
        if column in df.columns
    ]

    df["Total_Purchases"] = df[
        available_purchase_columns
    ].sum(axis=1)

    # --------------------------------------------------
    # Create Total Children
    # --------------------------------------------------

    if "Kidhome" in df.columns and "Teenhome" in df.columns:

        df["Total_Children"] = (
            df["Kidhome"] +
            df["Teenhome"]
        )

    # --------------------------------------------------
    # Create Campaign Response
    # --------------------------------------------------

    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response"
    ]

    available_campaign_columns = [
        column for column in campaign_columns
        if column in df.columns
    ]

    df["Total_Campaign_Response"] = df[
        available_campaign_columns
    ].sum(axis=1)

    return df


# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

def prepare_features(df):
    """
    Select clustering features and apply StandardScaler.
    """

    features = [
        "Age",
        "Income",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "NumDealsPurchases",
        "NumWebVisitsMonth"
    ]

    # Check whether all required features exist
    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    X = df[features].copy()

    # Handle any remaining missing values
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features,
        index=X.index
    )

    return X, X_scaled, scaler, features


# --------------------------------------------------
# Main Program
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CUSTOMER DATA PREPROCESSING")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Preprocess dataset
    df = preprocess_data(df)

    print("\nDataset Shape After Preprocessing:")
    print(df.shape)

    # Prepare clustering features
    X, X_scaled, scaler, features = prepare_features(df)

    print("\nFeatures Used for Clustering:")
    for feature in features:
        print("-", feature)

    print("\nScaled Feature Data:")
    print(X_scaled.head())

    print("\nPreprocessing completed successfully!")