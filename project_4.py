import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
from prophet import Prophet
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Q1: Load and clean the dataset
df = pd.read_csv("credit_card_transactions.csv")

df.info()
if "Unnamed: 0" in df.columns or "merch_zipcode" in df.columns:
    df.drop(["Unnamed: 0", "merch_zipcode"], axis=1, inplace=True)

df.head()

print(df.isnull().sum())

stats = df.describe()
stats
# ----------------

# Q2: Convert datetime columns
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

df["dob"] = pd.to_datetime(df["dob"])
df["customer_age"] = df["dob"].apply(lambda x: datetime.now().year - x.year)
# ----------------

# Q3: Calculate total spending per customer
df["total_spent"] = df.groupby("cc_num")["amt"].transform("sum")

df["transaction_count"] = df.groupby("cc_num")["amt"].transform("count")

df.head()
# ----------------

# Q4: Spending statistics per category
category_stats = df.groupby("category").agg(
    transaction_volume=("amt", "count"),
    total_amount_spent=("amt", "sum")
).sort_values(by="total_amount_spent", ascending=False)

category_stats.head()

plt.figure(figsize=(12, 6))
sns.barplot(x=category_stats.index, y=category_stats["total_amount_spent"], palette="viridis")
plt.xticks(rotation=45)
plt.title("Top Spending Categories by Total Amount")
plt.xlabel("Category")
plt.ylabel("Total Amount Spent ($)")
plt.savefig("Top Spending Categories by Total Amount.png")
plt.show()
# ----------------

# Q5: Aggregate daily spending trends
df["transaction_date"] = df["trans_date_trans_time"].dt.date
daily_spending = df.groupby("transaction_date")["amt"].sum()

plt.figure(figsize=(14, 6))
sns.lineplot(x=daily_spending.index, y=daily_spending.values, color="blue")
plt.title("Daily Spending Trends")
plt.xlabel("Date")
plt.ylabel("Total Amount Spent ($)")
plt.xticks(rotation=90)
plt.savefig("Daily Spending Trends.png")
plt.show()
# ----------------

# Q6: Spending trends by hour
df["transaction_hour"] = df["trans_date_trans_time"].dt.hour

hourly_spending = df.groupby("transaction_hour")["amt"].sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=hourly_spending.index, y=hourly_spending.values, palette="coolwarm")
plt.title("Peak Spending Hours in a Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Amount Spent ($)")
plt.xticks(range(0, 24))
plt.savefig("Peak Spending Hours in a Day.png")
plt.show()
# ----------------

# Q7: Top merchants by transaction volume
df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)
merchant_transactions = df["merchant"].value_counts().head(10)
merchant_transactions

plt.figure(figsize=(10, 5))
sns.barplot(y=merchant_transactions.index, x=merchant_transactions.values, palette="magma")
plt.title("Top Merchants by Transaction Volume")
plt.xlabel("Number of Transactions")
plt.ylabel("Merchant")
plt.savefig("Top Merchants by Transaction Volume.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q8: Define spending segments
df['spending_segment'] = pd.qcut(df['total_spent'], q=[0, 0.33, 0.66, 1], labels=['Low Spender', 'Medium Spender', 'High Spender'])

df['transaction_count'] = df.groupby("cc_num")["amt"].transform("count")

segment_analysis = df.groupby('spending_segment').agg({
    'amt': ['mean', 'median', 'max'],
    'transaction_count': 'mean',
    'category': lambda x: x.mode()[0]
})

segment_analysis

plt.figure(figsize=(8, 5))
sns.boxplot(x="spending_segment", y="amt", data=df, palette="coolwarm")
plt.title("Transaction Amount Distribution by Spending Segment")
plt.xlabel("Spending Segment")
plt.ylabel("Transaction Amount")
plt.savefig("Transaction Amount Distribution by Spending Segment.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q9: Average transactions per segment
avg_transactions = df.groupby("spending_segment")["transaction_count"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_transactions.index, y=avg_transactions.values, palette="coolwarm")
plt.title("Average Transaction Count per Spending Segment")
plt.xlabel("Spending Segment")
plt.ylabel("Average Transactions")
plt.savefig("Average Transaction Count per Spending Segment.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q10: Distribution of transaction amounts
plt.figure(figsize=(8, 5))
sns.histplot(df["amt"], bins=50, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Frequency")
plt.savefig("Distribution of Transaction Amounts.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["amt"], color="red")
plt.title("Boxplot of Transaction Amounts")
plt.xlabel("Transaction Amount ($)")
plt.savefig("Boxplot of Transaction Amounts.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df["amt"]), bins=50, kde=True, color="green")
plt.title("Log Transformed Distribution of Transaction Amounts")
plt.xlabel("Log(Transaction Amount + 1)")
plt.ylabel("Frequency")
plt.savefig("Log Transformed Distribution of Transaction Amounts.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q11: Correlation between spending behavior and location
city_spending = df.groupby("city")["amt"].sum().reset_index()
city_spending = city_spending.sort_values(by="amt", ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x="amt", y="city", data=city_spending, palette="viridis")
plt.xlabel("Total Spending ($)")
plt.ylabel("City")
plt.title("Top Cities by Total Spending")
plt.savefig("Top Cities by Total Spending.png", dpi=300, bbox_inches='tight')
plt.show()

state_spending = df.groupby("state")["amt"].sum().reset_index()
fig2 = px.choropleth(state_spending, locations='state', locationmode='USA-states', color='amt', title='Total Spending by State', color_continuous_scale='Viridis')
fig2.update_geos(scope="usa")
fig2.write_image("Total Spending by State.png", scale=2)

state_spending = df.groupby("state")["amt"].sum().reset_index()
state_spending = state_spending.sort_values(by="amt", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="amt", y="state", data=state_spending, palette="coolwarm")
plt.xlabel("Total Spending ($)")
plt.ylabel("State")
plt.title("Total Spending by State")
plt.xticks(rotation=90)
plt.savefig("Total Spending by State.png", dpi=300, bbox_inches='tight')
plt.show()

correlation = df[["city_pop", "amt"]].corr()
print("Correlation between City Population and Spending:\n", correlation)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap City Population vs Spending")
plt.savefig("Correlation Heatmap City Population vs Spending.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q12: Spending before and after payday
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["day_of_month"] = df["trans_date_trans_time"].dt.day

def categorize_payday(day):
    if day in [1, 2, 3, 4, 15, 16, 17, 18, 19]:
        return "After Payday"
    else:
        return "Before Payday"

df["payday_category"] = df["day_of_month"].apply(categorize_payday)

payday_spending = df.groupby("payday_category")["amt"].agg(["mean", "sum"]).reset_index()
payday_spending.columns = ["Payday Category", "Avg Transaction Amount", "Total Spending"]

payday_spending

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(ax=axes[0], x="Payday Category", y="Avg Transaction Amount", data=payday_spending, palette="coolwarm")
axes[0].set_title("Average Transaction Amount")
axes[0].set_ylabel("Avg Transaction Amount ($)")
axes[0].set_xlabel("Payday Category")

sns.barplot(ax=axes[1], x="Payday Category", y="Total Spending", data=payday_spending, palette="magma")
axes[1].set_title("Total Spending Before and After Payday")
axes[1].set_ylabel("Total Spending ($)")
axes[1].set_xlabel("Payday Category")

plt.tight_layout()
plt.savefig("Average Transaction Amount and Total Spending Before and After Payday.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q13: Detecting outliers in spending behavior
df["z_score"] = np.abs(zscore(df["amt"]))
df["outlier_z"] = df["z_score"] > 3

iso_forest = IsolationForest(contamination=0.01, random_state=42)
df["outlier_if"] = iso_forest.fit_predict(df[["amt"]]) == -1

plt.figure(figsize=(12, 6))
sns.boxplot(x=df["amt"], showfliers=True, color="red")
plt.title("Boxplot of Transaction Amounts Outliers in Red")
plt.xlabel("Transaction Amount ($)")
plt.savefig("Boxplot of Transaction Amounts Outliers in Red.png", dpi=300, bbox_inches='tight')
plt.show()
# ----------------

# Q14: Customer clustering using K-Means & DBSCAN
customer_spending = df.groupby("cc_num")["amt"].agg(["sum", "mean", "count"]).reset_index()
customer_spending.columns = ["Customer_ID", "Total_Spending", "Avg_Spending", "Transaction_Count"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_spending[["Total_Spending", "Avg_Spending", "Transaction_Count"]])

kmeans = KMeans(n_clusters=3, random_state=42)
customer_spending["KMeans_Cluster"] = kmeans.fit_predict(scaled_features)

dbscan = DBSCAN(eps=1, min_samples=5)
customer_spending["DBSCAN_Cluster"] = dbscan.fit_predict(scaled_features)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparison of K-Means and DBSCAN Customer Clustering", fontsize=14, fontweight="bold")

sns.scatterplot(ax=axes[0], data=customer_spending, x="Total_Spending", y="Avg_Spending", hue="KMeans_Cluster", palette="viridis", legend="full")
axes[0].set_title("K-Means Customer Clustering")
axes[0].set_xlabel("Total Spending ($)")
axes[0].set_ylabel("Average Spending ($)")

sns.scatterplot(ax=axes[1], data=customer_spending, x="Total_Spending", y="Avg_Spending", hue="DBSCAN_Cluster", palette="coolwarm", legend="full")
axes[1].set_title("DBSCAN Customer Clustering")
axes[1].set_xlabel("Total Spending ($)")
axes[1].set_ylabel("Average Spending ($)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Comparison of K Means and DBSCAN Customer Clustering.png", dpi=300, bbox_inches='tight')
plt.show()

fig = px.scatter_3d(customer_spending, x="Total_Spending", y="Avg_Spending", z="Transaction_Count", color="KMeans_Cluster", hover_data=["Customer_ID", "Total_Spending", "Avg_Spending", "Transaction_Count"], title="3D Scatter Plot of Customer Clusters")
fig.write_image("3D Scatter Plot of Customer Clusters.png", scale=2)
# ----------------

# Q15: Finding purchase patterns using Apriori
df_numeric = df.select_dtypes(include=['number'])
df_bool = df_numeric > 0

frequent_itemsets = apriori(df_bool, min_support=0.02, use_colnames=True)
frequent_itemsets = frequent_itemsets[frequent_itemsets["support"] > 0]

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules.dropna(inplace=True)

rules.sort_values(by="lift", ascending=False).head(10)
rules.to_csv("Association Rules.csv", index=False)
df.to_csv("Processed Data.csv", index=False)
# ----------------

# Q16: Handling skewness and outliers
from scipy.stats import skew
original_skewness = skew(df["amt"])
print("Skewness before log transformation:", original_skewness)

Q1 = df["amt"].quantile(0.25)
Q3 = df["amt"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["amt"] = np.where(df["amt"] < lower_bound, lower_bound, df["amt"])
df["amt"] = np.where(df["amt"] > upper_bound, upper_bound, df["amt"])

cleaned_skewness = skew(df["amt"])
print("Skewness after handling outliers:", cleaned_skewness)

df["amt"] = np.log1p(df["amt"])
transformed_skewness = skew(df["amt"])
print("Skewness after log transformation:", transformed_skewness)

df.head()
# ----------------

# Q17: Predicting future spending trends using Prophet
df_prophet = df.groupby("transaction_date")["amt"].sum().reset_index()
df_prophet.columns = ["ds", "y"]

model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=30, freq="D")
forecast = model.predict(future)

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

with open('Prophet Model.pkl', 'wb') as f:
    pickle.dump(model, f)

fig = model.plot(forecast)
plt.title("Transaction Amount Forecast")
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.savefig("Transaction Amount Forecast.png", dpi=300, bbox_inches="tight")
plt.show()

fig_components = model.plot_components(forecast)
fig_components.suptitle("Trend, Weekly, and Yearly Seasonality of Transactions", fontsize=14)
fig_components.savefig("Trend, Weekly, and Yearly Seasonality of Transactions.png", dpi=300, bbox_inches="tight")
plt.show()
# ----------------

# Q18: Anomaly detection using Prophet residuals
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
forecast["ds"] = pd.to_datetime(forecast["ds"])

df_prophet = df_prophet.merge(forecast[["ds", "yhat"]], on="ds", how="left")
df_prophet["residual"] = df_prophet["y"] - df_prophet["yhat"]

threshold = 3 * np.std(df_prophet["residual"])
df_prophet["anomaly"] = np.abs(df_prophet["residual"]) > threshold

plt.figure(figsize=(12, 6))
plt.plot(df_prophet["ds"], df_prophet["y"], label="Actual", color="blue", alpha=0.6)
plt.plot(df_prophet["ds"], df_prophet["yhat"], label="Forecast", color="green", linestyle="dashed")
anomalies = df_prophet[df_prophet["anomaly"]]
plt.scatter(anomalies["ds"], anomalies["y"], color="red", label="Anomaly", marker="o", s=50)
plt.xlabel("Date")
plt.ylabel("Transaction Amount")
plt.title("Transaction Amount Forecast with Anomalies")
plt.legend()
plt.savefig("Transaction Amount Forecast with Anomalies.png", dpi=300, bbox_inches="tight")
plt.show()
# ----------------

# Q19: Evaluate Prophet model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mae = mean_absolute_error(df_prophet["y"], df_prophet["yhat"])
print("Mean Absolute Error (MAE):", mae)

mse = mean_squared_error(df_prophet["y"], df_prophet["yhat"])
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

r2 = r2_score(df_prophet["y"], df_prophet["yhat"])
print("R-squared (R²):", r2)
