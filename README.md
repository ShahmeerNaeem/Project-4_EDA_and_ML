# 💳 Credit Card Transactions Analysis with Machine Learning & EDA

## 👉 Project Overview

This project analyzes credit card transactions to uncover spending patterns, detect outliers, segment customers, and identify potential fraud using Exploratory Data Analysis (EDA) and Machine Learning (ML) models. We leverage various statistical and visualization techniques to gain insights into transaction behaviors.

## 📚 Dataset

We use a credit card transaction dataset containing:

- **Transaction Details:** Transaction amount, date, merchant, and category.
- **Customer Information:** Age, spending habits, and transaction history.
- **Location Data:** City, state, and merchant location.

## 🔍 Exploratory Data Analysis (EDA)

### Key Questions Explored:

- What are the top spending categories and merchants?
- Are there specific hours or days with peak spending?
- How do customer spending habits vary across different regions?
- Do certain merchants show unusually high transaction volumes?
- What patterns exist in fraudulent vs. non-fraudulent transactions?

### 🔢 Key Insights:

- **Top Spending Categories:** Some categories consistently show higher transaction volumes.
- **Peak Spending Hours:** Spending peaks at specific times of the day.
- **Regional Trends:** Certain cities and states exhibit higher spending trends.
- **Spending Segments:** Customers can be grouped into **low, medium, and high spenders** based on transaction history.

## ⚙️ Machine Learning Pipeline

### 🔄 Data Processing:

- Handling missing values.
- Converting timestamps to meaningful features (hour, day, month).
- Encoding categorical data.
- Normalizing numerical variables.

### 🔧 Feature Engineering:

- Aggregating customer spending behavior.
- Creating spending segments.
- Identifying unusual transaction amounts.

### 📉 Clustering & Anomaly Detection:

- **K-Means & DBSCAN** to segment customers based on spending behavior.
- **Isolation Forest & Z-Score** for detecting outliers and potential fraudulent transactions.

### 📊 Association Rule Mining:

- **Apriori Algorithm** to identify frequently purchased items together.
- Extracting transaction patterns from high-spending customers.

## 💪 Results & Insights

- Customers exhibit distinct spending behaviors that can be used for targeted marketing.
- High-value transactions often follow seasonal trends.
- Machine Learning models successfully identify anomalies and clusters in spending behavior.

## 📊 Visualizations

- **Top Spending Categories & Merchants**
- **Daily & Hourly Spending Trends**
- **Customer Spending Segments**
- **Transaction Outlier Detection**
- **Clustering of Customers by Spending Habits**
- **Transaction Amount Distribution**
- **Fraud vs. Non-Fraud Transactions**
- **Geographical Heatmap of Transactions**
- **Merchant-wise Spending Trends**
- **Customer Spending Behavior Over Time**

## 🚀 Future Enhancements

- Implement a real-time fraud detection system.
- Enhance customer segmentation with deep learning.
- Integrate more external datasets for improved accuracy.

---

🌟 **Contributions Welcome!** Feel free to submit pull requests and report any issues.

🔗 **Repository:** [GitHub Link](https://github.com/your-repo/credit-card-analysis)

