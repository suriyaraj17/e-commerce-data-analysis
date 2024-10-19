import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('C:/Users/kotti/Downloads/archive (1).zip',encoding='ISO-8859-1')

# Display the first few rows of the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# Summary statistics
print(data.describe())
# Remove any rows where quantity is negative or zero
data = data[data['Quantity'] > 0]

# Remove rows where UnitPrice is negative or zero
data = data[data['UnitPrice'] > 0]

# Convert InvoiceDate to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Add a new column for TotalPrice
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Check the cleaned data
print(data.head())
#Top 10 most sold products
top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# Plot the top products
plt.figure(figsize=(10,6))
top_products.plot(kind='bar')
plt.title('Top 10 Most Sold Products')
plt.xlabel('Product Description')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.show()
# Group by month and calculate total sales
data['Month'] = data['InvoiceDate'].dt.month
monthly_sales = data.groupby('Month')['TotalPrice'].sum()

# Plot monthly sales
plt.figure(figsize=(10,6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()
# Customer distribution by country
customer_country = data['Country'].value_counts()

# Plot customer distribution
plt.figure(figsize=(10,6))
customer_country.plot(kind='bar')
plt.title('Customer Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()
# Calculate total revenue for each product
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
product_profit = data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)

# Plotting the most profitable products
plt.figure(figsize=(10,6))
product_profit.plot(kind='bar')
plt.title('Top 10 Most Profitable Products')
plt.xlabel('Product Description')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()
# Extract the day of the week
data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()

# Calculate total sales for each day of the week
sales_by_day = data.groupby('DayOfWeek')['TotalPrice'].sum()

# Plotting sales by day of the week
plt.figure(figsize=(10,6))
sales_by_day.plot(kind='bar')
plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()
from sklearn.cluster import KMeans

# Group by customer and aggregate features
customer_data = data.groupby('CustomerID').agg({
    'InvoiceNo': 'count',  # Frequency of purchases
    'TotalPrice': 'sum',   # Total revenue from the customer
}).rename(columns={'InvoiceNo': 'Frequency', 'TotalPrice': 'TotalSpent'})

# Apply K-Means clustering to segment customers
kmeans = KMeans(n_clusters=3)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Frequency', 'TotalSpent']])

# Visualize the clusters
plt.scatter(customer_data['Frequency'], customer_data['TotalSpent'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Frequency of Purchases')
plt.ylabel('Total Spent')
plt.show()