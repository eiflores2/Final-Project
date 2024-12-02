import pandas as pd
import nltk
# from nltk.corpus import stopwords
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import statistics
# from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("amazon.csv")

print(df.dtypes)


print(df.head())

# Display unique values in 'actual_price' column
# unique_values = df['actual_price'].unique()
# print(unique_values)



# numeric_df = df.select_dtypes(include=['number'])

df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace("'", "").str.replace(',', '')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

df = df.dropna(subset=['actual_price'])

# if df.empty:
#     print("DataFrame is empty after filering non_numeric values.")
# else:
#     print("DataFrame has numeric values in 'actual_price' column.")

descriptive_stats = df["actual_price"].describe()

additional_stats = df['actual_price'].agg(['mean', 'median', 'std', 'var','skew', 'kurt'])

print("Basic Descriptive Statistics: \n", descriptive_stats)
print("\nAdditional Statistics:\n", additional_stats)

plt.figure(figsize=(15,10))

plt.hist(df['actual_price'], bins=20)
plt.title("Histograms of Price")
plt.xlabel("Actual Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,8))
sns.boxplot(x=df["actual_price"])
plt.title("Box Plot of Price")
plt.show()