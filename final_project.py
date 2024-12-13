
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score

# We need a Function to read, clean, and optionally convert sales to From INR to USD
def read_and_clean_csv(file_path, sales_column, category_column, first_n_chars=3, convert_to_usd=False):
    data = pd.read_csv(file_path)
    data[sales_column] = data[sales_column].replace('[₹,]', '', regex=True).astype(float)
    if convert_to_usd:
        data[sales_column] = data[sales_column] * 0.012  # Correct INR to USD conversion rate
# We combined amazons columns into 3 columns Electronics, Home, and Music 
    if category_column in data.columns:
        category_mapping = {
            'Car': 'Ele', 'Com': 'Ele', 'Ele': 'Ele',
            'Hea': 'Hom', 'Hom': 'Hom',
            'Mus': 'Mus', 'Off': 'Mus', 'Toy': 'Mus'
        }
        data['Main Category'] = data[category_column].apply(lambda x: category_mapping.get(x[:first_n_chars], x[:first_n_chars]))
    elif 'Product Category' in data.columns:
        data['Main Category'] = data['Product Category']
    else:
        raise KeyError("No category column found")
    return data

# We need to use a Function to calculate total sales by category
def calculate_total_sales(data, category_column, sales_column):
    total_sales = data.groupby(category_column)[sales_column].sum().reset_index()
    return total_sales

# We decided to use a Function to calculate average ratings by category
def calculate_average_ratings(data, category_column, rating_column):
    if rating_column not in data.columns:
        raise KeyError(f"Column '{rating_column}' not found in the dataset.")
    
    # Since the ratings were considered a string, we needed to Convert ratings to numeric, handling non-numeric entries
    data[rating_column] = pd.to_numeric(data[rating_column], errors='coerce')
    
    # Group by category and calculate mean rating
    average_ratings = data.groupby(category_column)[rating_column].mean().reset_index()
    average_ratings.columns = [category_column, 'Average Rating']
    return average_ratings

# We used a Function to plot total sales by category using a bar chart
def plot_total_sales(total_sales_df, title):
    plt.figure(figsize=(12, 8))
    bars = plt.bar(total_sales_df.iloc[:, 0], total_sales_df.iloc[:, 1], color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom')

    plt.xlabel('Product Category')
    plt.ylabel('Total Sales (USD)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Similar to total sales, we used a def function to plot the average ratings of Amazon products using a bar chart
def plot_average_rating(average_rating_df, title):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=average_rating_df.iloc[:, 0], y=average_rating_df.iloc[:, 1], palette='viridis')

    for index, row in average_rating_df.iterrows():
        plt.text(index, row['Average Rating'], f'{row["Average Rating"]:.2f}', color='black', ha="center")

    plt.xlabel('Product Category')
    plt.ylabel('Average Rating')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# File paths
amazon_file = 'amazon.csv'
superstore_file = 'superstore.csv'

# Read, clean, and process the datasets
amazon_data = read_and_clean_csv(amazon_file, 'discounted_price', 'category', convert_to_usd=True)
superstore_data = read_and_clean_csv(superstore_file, 'Total Amount', 'Product Category')

amazon_average_rating = calculate_average_ratings(amazon_data, 'Main Category', 'rating')

print("Amazon Average Rating by Catehory:")
print(amazon_average_rating)

plot_average_rating(amazon_average_rating, 'Amazon Average Rating by Category')

# Check column names for superstore_data to find the transaction date column
print("Superstore Data Columns:", superstore_data.columns)

# Identify and process the date column
if 'Transaction Date' in superstore_data.columns:
    superstore_data['Transaction Date'] = pd.to_datetime(superstore_data['Transaction Date'])
elif 'Order Date' in superstore_data.columns:  # Example of an alternative column name
    superstore_data['Transaction Date'] = pd.to_datetime(superstore_data['Order Date'])
else:
    print("Transaction Date column not found. Adding a mock date column.")
    # Add a mock date column if no date column exists
    superstore_data['Transaction Date'] = pd.date_range(start='1/1/2023', periods=len(superstore_data), freq='D')

# Here, We Added a seasonality trend for Superstore to determine seasonality features
superstore_data['Month'] = superstore_data['Transaction Date'].dt.month
superstore_data['Day of Week'] = superstore_data['Transaction Date'].dt.day_name()

# Aggregate sales by category and month for Superstore
superstore_monthly_sales = superstore_data.groupby(['Main Category', 'Month'])['Total Amount'].sum().reset_index()

# Here, we implemented a Regression Model for Amazon
X_amazon = amazon_data[['discounted_price']]
y_amazon = amazon_data['discounted_price']
# To implement a regression model, we need to split our date into a train and test set
X_train_amazon, X_test_amazon, y_train_amazon, y_test_amazon = train_test_split(X_amazon, y_amazon, test_size=0.3, random_state=42)

# Once wr split our amazon dataset into a train and test set, we create a linear regression
reg_amazon = LinearRegression()
reg_amazon.fit(X_train_amazon, y_train_amazon)
y_pred_amazon = reg_amazon.predict(X_test_amazon)
mse_amazon = mean_squared_error(y_test_amazon, y_pred_amazon)
rmse_amazon = mse_amazon ** 0.5
print("Amazon Regression Results:")
print("RMSE:", rmse_amazon)
print("R²:", r2_score(y_test_amazon, y_pred_amazon))

# Similar to the Amazon dataset, we decided to implement a Regression Model for Superstore
X_superstore = superstore_monthly_sales[['Month']]
y_superstore = superstore_monthly_sales['Total Amount']
# To implement this, we need to split the superstore dataset into a train and test set
X_train_superstore, X_test_superstore, y_train_superstore, y_test_superstore = train_test_split(X_superstore, y_superstore, test_size=0.3, random_state=42)
# Once we split the dataset into a train and test set, it is time to build our linear regression
reg_superstore = LinearRegression()
reg_superstore.fit(X_train_superstore, y_train_superstore)
y_pred_superstore = reg_superstore.predict(X_test_superstore)
mse_superstore = mean_squared_error(y_test_superstore, y_pred_superstore)
rmse_superstore = mse_superstore ** 0.5
print("Superstore Regression Results:")
print("RMSE:", rmse_superstore)
print("R²:", r2_score(y_test_superstore, y_pred_superstore))

# At this point, we need to add a Classification Model for Amazon to determine the high performing and low performing contributors
amazon_data['Performance'] = amazon_data['discounted_price'].apply(
    lambda x: 'High' if x >= amazon_data['discounted_price'].median() else 'Low'
)

# Prepare the data for classification
X_amazon_class = amazon_data[['discounted_price']]
y_amazon_class = amazon_data['Performance']
X_train_amazon_class, X_test_amazon_class, y_train_amazon_class, y_test_amazon_class = train_test_split(
    X_amazon_class, y_amazon_class, test_size=0.3, random_state=42
)


#Train the random forest classifier
clf_amazon = RandomForestClassifier(random_state=42)
clf_amazon.fit(X_train_amazon_class, y_train_amazon_class)
y_pred_amazon_class = clf_amazon.predict(X_test_amazon_class)
print("Amazon Classification Results:")
print(classification_report(y_test_amazon_class, y_pred_amazon_class))
print("Accuracy:", accuracy_score(y_test_amazon_class, y_pred_amazon_class))

# Visualize Superstore Seasonality Trends
sns.lineplot(data=superstore_monthly_sales, x='Month', y='Total Amount', hue='Main Category')
plt.title('Superstore Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# Plot total sales
plot_total_sales(calculate_total_sales(amazon_data, 'Main Category', 'discounted_price'), 'Amazon Total Sales by Category')
plot_total_sales(superstore_monthly_sales.groupby('Main Category')['Total Amount'].sum().reset_index(), 'Superstore Total Sales by Category')


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html?form=MG0AV3
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?form=MG0AV3
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html?form=MG0AV3
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html?form=MG0AV3
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html?form=MG0AV3


# dataset
# Superstore Dataset:
# The data was extracted from the Kaggle Dataset(https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset.) This dataset was obtained for educational purposes by Eduardo I. Flores and Luis Vargas
# Amazon Sales Dataset:
# The dataset was extracted from the Kaggle Dataset (https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset.) This dataset was obtained for educational purposes by Eduardo I. Flores and Luis Vargas


# Import Libraries: The code imports several Python libraries:

# pandas: Used for data manipulation and analysis.
# matplotlib.pyplot and seaborn: Used for creating visualizations.
# sklearn: Used for splitting data into training and testing sets, performing regression, and classification.
# Read and Clean Data:
# The read_and_clean_csv function reads a CSV file, cleans the sales data by removing currency symbols, and optionally converts sales to USD. It also maps detailed product categories to broader categories.
# Calculate Total Sales:
# The calculate_total_sales function groups the sales data by category and calculates the total sales for each category.
# Calculate Average Ratings:
# The calculate_average_ratings function calculates the average rating for each product category, converting non-numeric ratings to numeric and handling any errors.
# Plot Total Sales:
# The plot_total_sales function creates a bar plot showing the total sales for each product category.
# Plot Average Ratings:
# The plot_average_rating function creates a bar plot showing the average rating for each product category.
# Load and Process Data:
# The script reads and cleans data from two CSV files (Amazon and Superstore), processes the data, and calculates average ratings for the Amazon data.
# Print and Plot Average Ratings:
# The average ratings for Amazon categories are printed and plotted.
# Process Date Columns for Superstore Data:
# The script checks for date columns in the Superstore data, converts them to datetime format, and adds a mock date column if none are found.
# Add Seasonality Features:
# The script adds month and day-of-week features to the Superstore data to account for seasonality.
# Aggregate Monthly Sales for Superstore:
# The script groups the Superstore data by category and month, calculating the total sales for each combination.
# Regression Models:
# Amazon: The script trains a linear regression model on the Amazon data to predict future sales based on the discounted price.
# Superstore: Similarly, it trains a linear regression model on the Superstore data to predict future sales based on the month.
# Print Regression Results:
# The script calculates and prints the Root Mean Squared Error (RMSE) and R² score for both regression models.
# Classification Model for Amazon:
# The script adds a Performance column to the Amazon data, classifying products as "High" or "Low" based on the median discounted price.
# It then trains a random forest classifier to predict the performance category of the products.
# The script prints the classification results, including a classification report and accuracy score.
# Visualize Superstore Sales Trends:
# The script creates a line plot showing the monthly sales trends for Superstore categories.
# Plot Total Sales:
# The script plots the total sales for each category from both Amazon and Superstore data.
# Overall, the script performs data cleaning, sales and ratings analysis, regression and classification modeling, and visualizes the results to compare sales performance between Amazon and Superstore and predict future sales trends.
# If you have any more questions or need further clarifications, let me know!

