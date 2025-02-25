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
- Are there correlations between transaction frequency and fraud probability?
- How do spending patterns change over time for different customer segments?

### 🔢 Key Insights:

- **Top Spending Categories:** Some categories consistently show higher transaction volumes.
- **Peak Spending Hours:** Spending peaks at specific times of the day.
- **Regional Trends:** Certain cities and states exhibit higher spending trends.
- **Spending Segments:** Customers can be grouped into **low, medium, and high spenders** based on transaction history.
- **Fraudulent Transactions:** Fraudulent transactions often occur at unusual hours and involve abnormally high amounts.
- **High-Risk Merchants:** Some merchants have disproportionately high fraud rates compared to others.
- **Transaction Frequency & Amount Relationship:** Frequent low-value transactions are more common among legitimate users, whereas sporadic high-value transactions are often flagged as fraudulent.

## ⚙️ Machine Learning Pipeline

### 🔄 Data Processing:

- Handling missing values through imputation techniques.
- Converting timestamps to meaningful features (hour, day, month, weekday/weekend).
- Encoding categorical variables using one-hot encoding and label encoding.
- Normalizing numerical variables to ensure uniform scaling.
- Removing duplicate transactions and extreme outliers.
- Creating fraud labels and balancing the dataset for better model performance.

### 🔧 Feature Engineering:

- Creating new features based on historical spending behavior.
- Categorizing transactions based on frequency and amount.
- Computing rolling averages to detect spending trends.
- Assigning risk scores to transactions based on past fraudulent activities.
- Extracting location-based spending patterns.
- Creating merchant-specific risk profiles based on past fraudulent activity.
- Identifying repeated transactions at the same merchant within a short timeframe as potential fraud indicators.

### 📉 Clustering & Anomaly Detection:

- **K-Means Clustering:** Grouping customers based on transaction frequency and spending habits.
- **DBSCAN:** Identifying noise and outliers in customer transaction patterns.
- **Isolation Forest:** Detecting fraudulent transactions using anomaly scores.
- **Z-Score Analysis:** Identifying unusually high transaction amounts.
- **Local Outlier Factor (LOF):** Detecting transactions that deviate significantly from normal spending behavior.
- **Autoencoders (Deep Learning):** Identifying anomalies based on feature reconstruction errors.

### 🤖 Supervised Machine Learning Models:

- **Logistic Regression:** Predicting fraudulent transactions based on transaction characteristics.
- **Random Forest:** Classifying transactions as fraudulent or non-fraudulent with feature importance analysis.
- **XGBoost:** Optimized fraud detection with high precision and recall.
- **Neural Networks (Optional):** Deep learning-based approach for advanced fraud detection.
- **Support Vector Machines (SVM):** Identifying fraud cases in high-dimensional feature spaces.
- **Ensemble Methods:** Combining multiple models to improve fraud detection accuracy.

### 📊 Association Rule Mining:

- **Apriori Algorithm:** Identifying frequently purchased items together.
- **Market Basket Analysis:** Understanding co-occurrence of spending patterns across different merchant categories.

## 💪 Results & Insights

- Customers exhibit distinct spending behaviors that can be used for targeted marketing.
- High-value transactions often follow seasonal trends.
- Machine Learning models successfully identify anomalies and clusters in spending behavior.
- Fraudulent transactions tend to cluster around specific merchants and time frames.
- Customers who frequently shop at multiple high-end merchants have a higher risk of fraudulent transactions.
- **Fraudulent transactions often involve international merchants or unusual locations.**
- **Repeat transactions at the same merchant within short timeframes may indicate fraudulent behavior.**
- **Fraud detection models show high precision, but further optimization is needed to reduce false positives.**
- **Anomaly detection methods successfully flagged outliers, but a hybrid approach improves accuracy.**
- **Feature importance analysis indicates that transaction amount, merchant category, and transaction time are key fraud indicators.**

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
- **Anomaly Scores Distribution for Fraud Detection**
- **Feature Importance in Fraud Classification**
- **Comparison of Different Fraud Detection Models**
- **Spending Trends for Different Customer Segments**
- **Time-Series Analysis of Fraudulent Transactions**
- **Correlation Heatmap of Key Transaction Features**
- **Merchant Risk Score Distribution**

## 🚀 Future Enhancements

- Implement a real-time fraud detection system.
- Enhance customer segmentation with deep learning.
- Integrate more external datasets for improved accuracy.
- Use Reinforcement Learning for adaptive fraud detection models.
- Develop a mobile-friendly dashboard for real-time analytics.
- Improve fraud detection recall by reducing false positives through better feature selection.
- Implement adversarial training methods to detect evolving fraud strategies.
- Leverage blockchain technology for enhanced transaction security.

---

🌟 **Contributions Welcome!** Feel free to submit pull requests and report any issues.

🔗 **Repository:** [GitHub Link](https://github.com/your-repo/credit-card-analysis)

