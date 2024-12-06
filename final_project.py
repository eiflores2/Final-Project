import pandas as pd
import nltk
# from nltk.corpus import stopwords
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import statistics
# from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("amazon.csv")

def read_and_clean_csv(file_path, sales_column, category_column, first_n_chars=3, convert_to_usd=False):
    data = pd.read_csv(file_path)
    data[sales_column] = data[sales_column].replace('[₹,]', '', regex=True).astype(float)
    if convert_to_usd:
        data[sales_column] = data[sales_column] * 0.12

    if category_column in data.columns:
        data['Main Category'] = data[category_column].apply(lambda x: x[:first_n_chars] if isinstance(x, str) else x)
    elif 'Product Category' in data.columns:
        data ['Main Category'] = data['Product Category']
    else:
        raise KeyError("No category column found")
    return data

def calculate_total_sales(data, category_column, sales_column):
    total_sales = data.groupby(category_column)[sales_column].sum().reset_index()
    return total_sales

def plot_total_sales(total_sales_df, title):
    plt.figure(figsize=(12,8))
    bars = plt.bar(total_sales_df.iloc[:, 0], total_sales_df.iloc[:, 1], color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom')

    plt.xlabel('Product Category')
    plt.ylabel('Total Sales (USD)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

amazon_file = 'amazon.csv'
superstore_file = 'superstore.csv'

amazon_data = read_and_clean_csv(amazon_file,'discounted_price', 'category', convert_to_usd=True)
superstore_data = read_and_clean_csv(superstore_file, 'Total Amount', 'Product Category')

amazon_total_sales = calculate_total_sales(amazon_data, 'Main Category', 'discounted_price')
superstore_total_sales = calculate_total_sales(superstore_data, 'Main Category', 'Total Amount')

print("Amazon Total Sales by Category(FIrst 3 Letter):")
print(amazon_total_sales)

print("\nSuperstore Total Sales by Category:")
print(superstore_total_sales)

plot_total_sales(amazon_total_sales, 'Amazon Total Sales by Category(First 3 Letters)')

plot_total_sales(superstore_total_sales, 'Superstore Total Sales by Category')