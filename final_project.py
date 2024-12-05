import pandas as pd
import nltk
# from nltk.corpus import stopwords
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import statistics
# from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("amazon.csv")

def read_and_clean_csv(file_path, sales_column, category_column, first_n_chars=3):
    data = pd.read_csv(file_path)
    data[sales_column] = data[sales_column].replace('[₹,]', '', regex=True).astype(float)

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
    plt.ylabel('Total Sales')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

amazon_file = 'amazon.csv'
superstore_file = 'superstore.csv'

amazon_data = read_and_clean_csv(amazon_file,'discounted_price', 'category')
superstore_data = read_and_clean_csv(superstore_file, 'Total Amount', 'Product Category')

amazon_total_sales = calculate_total_sales(amazon_data, 'Main Category', 'discounted_price')
superstore_total_sales = calculate_total_sales(superstore_data, 'Main Category', 'Total Amount')

print("Amazon Total Sales by Category(FIrst 3 Letter):")
print(amazon_total_sales)

print("\nSuperstore Total Sales by Category:")
print(superstore_total_sales)

plot_total_sales(amazon_total_sales, 'Amazon Total Sales by Category(First 3 Letters)')

plot_total_sales(superstore_total_sales, 'Superstore Total Sales by Category')

# def compare_sales(file1,file2):
#     data1 = read_csv(file1)
#     data2 = read_csv(file2)

#     data1 = clean_sales_values(data1, 'discounted_price')
#     data2 = clean_sales_values(data2, 'Total Amount')

#     categories = set(data1['category']).union(set(data2['Product Category']))
#     comparison = []

#     for category in categories:
#         sales1 = data1[data1['category'] == category]['discounted_price'].sum() if category in data1['category'].values else 0
#         sales2 = data2[data2['Product Category'] == category]['Total Amount'].sum() if category in data2['Product Category'].values else 0
#         comparison.append({'Product Category': category, 'Amazon': sales1, 'Superstore': sales2})

#     return pd.DataFrame(comparison)

# def plot_comparison(comparison_df):
#     bar_width = 0.35
#     index = range(len(comparison_df))

#     fig, ax = plt.subplots(figsize=(12,8))

#     bar1 = ax.bar(index, comparison_df['Amazon'], bar_width, label='Amazon')
#     bar2 = ax.bar([i + bar_width for i in index], comparison_df['Superstore'], bar_width, label='Superstore')


#     ax.set_xlabel('Prodcut Category')
#     ax.set_ylabel('Sales')
#     ax.set_title('Sales Comparison by Product Category') 
#     ax.set_xticks([i + bar_width / 2 for i in index])
#     ax.set_xticklabels(comparison_df['Product Category'], rotation=45) 
#     ax.legend()

#     plt.tight_layout()
#     plt.show() 

# file1 = 'amazon.csv'
# file2 = 'superstore.csv'
# comparison_df = compare_sales(file1, file2)
# print(comparison_df)

# plot_comparison(comparison_df)

# print(df.dtypes)


# print(df.head())

# Display unique values in 'actual_price' column
# unique_values = df['actual_price'].unique()
# print(unique_values)



# numeric_df = df.select_dtypes(include=['number'])

# df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace("'", "").str.replace(',', '')
# df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

# df = df.dropna(subset=['actual_price'])

# # if df.empty:
# #     print("DataFrame is empty after filering non_numeric values.")
# # else:
# #     print("DataFrame has numeric values in 'actual_price' column.")

# descriptive_stats = df["actual_price"].describe()

# additional_stats = df['actual_price'].agg(['mean', 'median', 'std', 'var','skew', 'kurt'])

# print("Basic Descriptive Statistics: \n", descriptive_stats)
# print("\nAdditional Statistics:\n", additional_stats)

# plt.figure(figsize=(15,10))

# plt.hist(df['actual_price'], bins=20)
# plt.title("Histograms of Price")
# plt.xlabel("Actual Price")
# plt.ylabel("Frequency")
# plt.show()

# plt.figure(figsize=(10,8))
# sns.boxplot(x=df["actual_price"])
# plt.title("Box Plot of Price")
# plt.show()