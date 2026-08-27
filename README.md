# customer_segmentation
# Customer Segmentation using K-Means Clustering

## Project Overview

Customer Segmentation is a machine learning project that groups customers into different segments based on their characteristics and purchasing behavior.

This project uses the **K-Means Clustering algorithm**, an unsupervised machine learning technique, to identify groups of customers with similar behavior.

The project analyzes factors such as customer age, income, recency, total spending, total purchases, deal purchases, and website visits.

The resulting customer segments can help businesses understand their customers and develop more effective marketing strategies.

## Objectives

* Group customers based on similar characteristics.
* Identify different types of customers.
* Identify high-value and potential customers.
* Understand customer purchasing behavior.
* Improve targeted marketing strategies.
* Support data-driven business decisions.
* Improve customer satisfaction and business efficiency.

## Problem Statement

Businesses may have a large number of customers with different purchasing behaviors. Treating every customer in the same way can result in inefficient marketing and poor resource utilization.

Businesses need a way to identify groups of customers with similar characteristics and purchasing patterns.

This project solves this problem by applying K-Means Clustering to customer data and dividing customers into meaningful groups.

## Proposed Solution

The system performs the following steps:

1. Load the customer dataset.
2. Clean and preprocess the data.
3. Create useful customer features.
4. Select relevant features for clustering.
5. Scale the selected features.
6. Determine the appropriate number of clusters.
7. Apply the K-Means Clustering algorithm.
8. Assign each customer to a cluster.
9. Analyze the characteristics of each cluster.
10. Generate customer segmentation results.
11. Provide business insights based on the identified segments.

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

### Development Tools

* Jupyter Notebook
* Visual Studio Code or any Python IDE
* Command Prompt / Terminal
* GitHub

## Dataset

The project uses a customer marketing dataset containing **2,240 customer records and 29 original features**.

Important original features include:

* ID
* Year_Birth
* Education
* Marital_Status
* Income
* Kidhome
* Teenhome
* Recency
* MntWines
* MntFruits
* MntMeatProducts
* MntFishProducts
* MntSweetProducts
* MntGoldProds
* NumDealsPurchases
* NumWebPurchases
* NumCatalogPurchases
* NumStorePurchases
* NumWebVisitsMonth
* AcceptedCmp1 to AcceptedCmp5
* Response

Additional features are created during preprocessing.

## Features Used for Clustering

The following features are used by the K-Means model:

* Age
* Income
* Recency
* Total_Spending
* Total_Purchases
* NumDealsPurchases
* NumWebVisitsMonth

### Feature Description

| Feature           | Description                                       |
| ----------------- | ------------------------------------------------- |
| Age               | Age of the customer                               |
| Income            | Customer's annual income                          |
| Recency           | Number of days since the customer's last purchase |
| Total_Spending    | Total amount spent across product categories      |
| Total_Purchases   | Total number of web, catalog, and store purchases |
| NumDealsPurchases | Number of purchases made using deals              |
| NumWebVisitsMonth | Number of website visits per month                |

`Customer ID` is not used as a clustering feature because it is an identifier rather than a meaningful behavioral characteristic.

## Project Workflow

```text
Customer Dataset
       |
       v
Data Loading
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Feature Selection
       |
       v
Feature Scaling
       |
       v
Finding Optimal K
       |
       v
K-Means Clustering
       |
       v
Cluster Assignment
       |
       v
Cluster Analysis
       |
       v
Customer Segmentation
       |
       v
Business Insights
       |
       v
Prediction / Deployment
```

## Data Preprocessing

The preprocessing stage includes:

* Loading the dataset.
* Removing duplicate records.
* Handling missing income values.
* Converting customer date information.
* Creating the Age feature.
* Removing unrealistic age values.
* Creating Total_Spending.
* Creating Total_Purchases.
* Creating Total_Children.
* Creating Total_Campaign_Response.
* Handling remaining missing values.
* Scaling numerical features using StandardScaler.

## Feature Engineering

### Age

Age is calculated from the `Year_Birth` column.

```text
Age = Current Year - Year_Birth
```

### Total Spending

Total spending is calculated from:

```text
MntWines
MntFruits
MntMeatProducts
MntFishProducts
MntSweetProducts
MntGoldProds
```

### Total Purchases

Total purchases are calculated from:

```text
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
```

## K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data into K groups based on similarity.

The algorithm works through the following steps:

1. Select the number of clusters K.
2. Initialize cluster centroids.
3. Assign each customer to the nearest centroid.
4. Calculate new centroids.
5. Repeat the assignment and centroid calculation process.
6. Stop when the clusters converge.

## Finding the Optimal Number of Clusters

Two techniques are used to evaluate the number of clusters.

### Elbow Method

The Elbow Method calculates the Within-Cluster Sum of Squares (WCSS) for different values of K.

The WCSS measures how close data points are to their assigned cluster centers.

The value of K is selected by examining the point where the decrease in WCSS begins to slow down significantly.

### Silhouette Score

The Silhouette Score evaluates how well each customer fits within its assigned cluster compared with other clusters.

A higher Silhouette Score generally indicates better-separated and more meaningful clusters.

The project evaluates K values from 2 to 10 and selects the K with the best silhouette score.

## Customer Segmentation

After training the K-Means model, every customer receives a cluster number.

The clusters are then analyzed using:

* Average income
* Average spending
* Average purchases
* Average age
* Recency
* Deal purchases
* Website visits

The cluster characteristics are used to give meaningful business descriptions to the customer groups.

Possible segment names include:

* Premium Customers
* Budget Customers
* Regular Customers
* Potential Customers

The exact interpretation depends on the characteristics of the clusters produced by the trained model.

## Visualization

The project includes visualizations such as:

* Age distribution
* Income distribution
* Total spending distribution
* Income vs Total Spending
* Age vs Total Spending
* Income vs Total Purchases
* Correlation heatmap
* Elbow curve
* Silhouette score comparison
* Customer cluster visualization
* Cluster spending comparison
* Cluster income comparison
* Cluster purchase comparison

## Project Files

All project files are kept in a single folder.

```text
Customer_Segmentation/
│
├── customer_segmentation.csv
├── customer_segmentation.ipynb
├── preprocess.py
├── train_model.py
├── predict.py
├── app.py
├── model.pkl
├── customer_segmentation_results.csv
├── requirements.txt
└── README.md
```

## File Description

### customer_segmentation.csv

Original customer dataset used for the project.

### customer_segmentation.ipynb

Jupyter Notebook containing:

* Data loading
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Feature selection
* Elbow Method
* Silhouette Score
* K-Means clustering
* Visualization
* Cluster analysis
* Business insights

### preprocess.py

Preprocesses the customer dataset and prepares the features required by the clustering model.

### train_model.py

Trains the K-Means clustering model, evaluates different cluster counts using the Silhouette Score, selects the optimal K, and saves the trained model.

### predict.py

Takes customer information as input and predicts the customer's cluster.

### app.py

Provides an interactive Streamlit application for customer segmentation.

### model.pkl

Stores the trained K-Means model, StandardScaler, selected features, and optimal number of clusters.

### customer_segmentation_results.csv

Contains the customer dataset with the assigned cluster information.

### requirements.txt

Contains the Python libraries required to run the project.

## How to Run the Project

### Step 1: Install Python

Make sure Python is installed on your computer.

Check the installation using:

```bash
python --version
```

### Step 2: Install Required Libraries

Open Command Prompt or Terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### Step 3: Run Preprocessing

```bash
python preprocess.py
```

### Step 4: Train the Model

```bash
python train_model.py
```

This creates:

```text
model.pkl
customer_segmentation_results.csv
```

### Step 5: Make a Prediction

Run:

```bash
python predict.py
```

The program will ask for customer information and return the predicted customer cluster.

### Step 6: Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The Customer Segmentation System will open in a web browser.

## Expected Output

The project produces:

* Customer clusters.
* Cluster visualization.
* Cluster profiles.
* Customer segment analysis.
* A segmented customer CSV file.
* A trained K-Means model.
* Individual customer cluster predictions.
* An interactive Streamlit application.

## Business Benefits

Customer segmentation can help businesses:

* Identify high-value customers.
* Understand customer purchasing behavior.
* Create targeted marketing campaigns.
* Provide personalized offers.
* Improve customer retention.
* Reduce unnecessary marketing costs.
* Increase sales and profitability.
* Improve customer satisfaction.

## Limitations

* K-Means requires selecting an appropriate number of clusters.
* The algorithm can be sensitive to outliers.
* Results depend on the quality of the dataset.
* Different feature selections can produce different clusters.
* Cluster numbers themselves do not have business meaning.
* Customer segments may change as new customer data becomes available.

## Future Enhancements

The project can be improved by:

* Implementing DBSCAN clustering.
* Implementing Hierarchical Clustering.
* Comparing multiple clustering algorithms.
* Adding real-time customer segmentation.
* Integrating the system with CRM software.
* Adding customer lifetime value analysis.
* Developing personalized product recommendations.
* Using advanced machine learning techniques.
* Connecting the application to a live database.

## Conclusion

The Customer Segmentation project demonstrates how machine learning can be used to group customers according to their characteristics and purchasing behavior.

K-Means Clustering provides an effective approach for discovering customer groups without requiring predefined labels.

By analyzing income, spending, purchasing behavior, recency, and online activity, businesses can better understand their customers and develop targeted marketing strategies.

The project combines data preprocessing, feature engineering, exploratory data analysis, clustering, visualization, and deployment into a complete customer segmentation system.
