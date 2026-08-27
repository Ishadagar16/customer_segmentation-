import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocess import load_data, preprocess_data, prepare_features


# --------------------------------------------------
# Load and Preprocess Dataset
# --------------------------------------------------

print("=" * 60)
print("CUSTOMER SEGMENTATION - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = load_data("customer_segmentation.csv")

print("Original dataset shape:", df.shape)


print("\nPreprocessing dataset...")

df = preprocess_data(df)

print("Preprocessed dataset shape:", df.shape)


# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

print("\nPreparing features...")

X, X_scaled, scaler, features = prepare_features(df)

print("\nFeatures used for clustering:")

for feature in features:
    print("-", feature)


# --------------------------------------------------
# Find Optimal Number of Clusters
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

silhouette_scores = {}

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# --------------------------------------------------
# Select Best K
# --------------------------------------------------

optimal_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

best_score = silhouette_scores[optimal_k]

print("\nOptimal number of clusters:", optimal_k)
print("Best Silhouette Score:", round(best_score, 4))


# --------------------------------------------------
# Train Final K-Means Model
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING FINAL K-MEANS MODEL")
print("=" * 60)

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

print("K-Means model trained successfully!")


# --------------------------------------------------
# Add Cluster Labels
# --------------------------------------------------

df["Cluster"] = clusters

print("\nCustomer count in each cluster:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# Display Cluster Profile
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER PROFILE")
print("=" * 60)

cluster_profile = df.groupby("Cluster")[
    [
        "Age",
        "Income",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "NumDealsPurchases",
        "NumWebVisitsMonth"
    ]
].mean().round(2)

print(cluster_profile)


# --------------------------------------------------
# Save Clustered Dataset
# --------------------------------------------------

df.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

print(
    "\nClustered dataset saved as "
    "customer_segmentation_results.csv"
)


# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_data = {
    "model": kmeans,
    "scaler": scaler,
    "features": features,
    "optimal_k": optimal_k
}

with open("model.pkl", "wb") as file:

    pickle.dump(
        model_data,
        file
    )

print("\nModel saved successfully as model.pkl")


# --------------------------------------------------
# Training Completed
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)