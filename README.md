import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

# Load the data
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Select relevant columns
sold = sold[["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CLOSEDATE", "CITY", "COUNTY", "MHIA21", "AHIA21", "ACRES", "STYLE", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "BATHSTOTAL", "VACANTA21", "HHA21"]]

# Update 'PROPTYPE' based on 'STYLE'
sold.loc[sold['STYLE'].str.contains('Townhouse', na=False), 'PROPTYPE'] = 'TH'
sold.loc[sold['STYLE'].str.contains('Mobile Home', na=False), 'PROPTYPE'] = 'MH'
sold.loc[sold['STYLE'].str.contains('Cape Cod', na=False), 'PROPTYPE'] = 'SC'

sold['VACANCY'] = sold["VACANTA21"] / sold["HHA21"]

# Convert 'CLOSEDATE' to datetime format and extract the year
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold['CLOSEYEAR'] = sold['CLOSEDATE'].dt.year.fillna(0).astype(int)

# Create P_SQFT
sold['P_SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]
    
sold["INCOME"] = sold.apply(choose_income, axis=1)

tax_bins = [
    -np.inf, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 
    5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 
    13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 21000, 22000, 
    23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 
    33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 
    43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, np.inf
]

tax_bin_labels = [
    '<500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', 
    '3000-3500', '3500-4000', '4000-4500', '4500-5000', '5000-5500', 
    '5500-6000', '6000-6500', '6500-7000', '7000-7500', '7500-8000', 
    '8000-8500', '8500-9000', '9000-9500', '9500-10000', '10000-10500', 
    '10500-11000', '11000-11500', '11500-12000', '12000-12500', '12500-13000', 
    '13000-13500', '13500-14000', '14000-14500', '14500-15000', '15000-15500', 
    '15500-16000', '16000-16500', '16500-17000', '17000-17500', '17500-18000', 
    '18000-18500', '18500-19000', '19000-19500', '19500-20000', '20000-21000', 
    '21000-22000', '22000-23000', '23000-24000', '24000-25000', '25000-26000', 
    '26000-27000', '27000-28000', '28000-29000', '29000-30000', '30000-31000', 
    '31000-32000', '32000-33000', '33000-34000', '34000-35000', '35000-36000', 
    '36000-37000', '37000-38000', '38000-39000', '39000-40000', '40000-41000', 
    '41000-42000', '42000-43000', '43000-44000', '44000-45000', '45000-46000', 
    '46000-47000', '47000-48000', '48000-49000', '49000-50000', '>50000'
]

# Create a new column 'TAX_BIN' with the binned data
sold['TAX_BIN'] = pd.cut(sold['PROPTAX'], bins=tax_bins, labels=tax_bin_labels)

# Group by 'TAX_BIN' and include additional columns using first()
grouped = sold.groupby(['CLOSEYEAR', 'CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN'], observed=True).agg(
    median_closeprice=('CLOSEPRICE', 'median'),
    max_closeprice = ('CLOSEPRICE', 'max'),
    min_closeprice =('CLOSEPRICE', 'min'),
    count=('CLOSEPRICE', 'size'),
    ABGSQFT=('ABGSQFT', 'median'),
    TOT_SQFT=('TOT_SQFT', 'median'),
    ACRES=('ACRES', 'median'),
    INCOME=('INCOME', 'median'),
    YEAR_BUILT=("YEARBUILT", 'median')
).reset_index()

# Filter groups where the count is at least 2
filtered = grouped[grouped['count'] >= 2]

# Sort the filtered DataFrame by the grouping columns and CLOSEYEAR
filtered = filtered.sort_values(by=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'])

# Calculate the year-over-year changes for median_closeprice
filtered['median_closeprice_yoy_change'] = filtered.groupby(['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN'], observed=True)['median_closeprice'].pct_change()

# Encoding categorical features
label_encoders = {}
categorical_cols = ['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN']

for col in categorical_cols:
    le = LabelEncoder()
    filtered[col] = le.fit_transform(filtered[col].astype(str))
    label_encoders[col] = le

# Separate data into features and target
features = filtered[['CLOSEYEAR', 'CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'ABGSQFT', 'TOT_SQFT', 'ACRES', 'INCOME', 'YEAR_BUILT']]
target = filtered['median_closeprice_yoy_change']

# Split the data into training and prediction sets
train_data = filtered[filtered['median_closeprice_yoy_change'].notna()]
predict_data = filtered[filtered['median_closeprice_yoy_change'].isna()]

X_train = train_data[features.columns]
y_train = train_data['median_closeprice_yoy_change']
X_predict = predict_data[features.columns]

# Train a RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the missing values
predicted_yoy_change = model.predict(X_predict)

# Fill in the predicted values in the original dataframe
filtered.loc[filtered['median_closeprice_yoy_change'].isna(), 'median_closeprice_yoy_change'] = predicted_yoy_change

# Decode the encoded categorical columns back to their original values
for col in categorical_cols:
    le = label_encoders[col]
    filtered.loc[:, col] = le.inverse_transform(filtered[col].astype(int))

# Save the updated dataframe to a new CSV file
filtered.to_csv(r'C:\cctaddr\RES_SOLD_SORTED_UPDATED.csv', index=False)

print(f"Updated file saved to {r'C:\cctaddr\RES_SOLD_SORTED_UPDATED.csv'}")

# Encoding categorical features again for the next steps
for col in categorical_cols:
    le = LabelEncoder()
    filtered.loc[:, col] = le.fit_transform(filtered[col].astype(str))
    label_encoders[col] = le

# Separate data into features and target
features = filtered[['CLOSEYEAR', 'CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'ABGSQFT', 'TOT_SQFT', 'ACRES', 'INCOME', 'YEAR_BUILT']]
target = filtered['median_closeprice']

# Create a combination of all unique values for CITY, PROPTYPE, DIRECT, and TAX_BIN
unique_combinations = filtered[['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN']].drop_duplicates()

# Determine the range of years in the dataset
min_year = int(filtered['CLOSEYEAR'].min())
max_year = int(filtered['CLOSEYEAR'].max())

# Create a DataFrame with all combinations of unique values and years
all_years = pd.DataFrame({'CLOSEYEAR': range(min_year, max_year + 1)})
all_combinations = unique_combinations.merge(all_years, how='cross')

# Merge with the original dataframe to identify missing combinations
merged_filtered = all_combinations.merge(filtered, on=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'], how='left')

# Identify rows with missing median_closeprice
missing_groups = merged_filtered[merged_filtered['median_closeprice'].isna()]

# Prepare training data using existing non-missing values
train_data = filtered[filtered['median_closeprice'].notna()]
predict_data = missing_groups

X_train = train_data[features.columns]
y_train = train_data['median_closeprice']
X_predict = predict_data[features.columns]

# Impute missing values in the feature set
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_predict_imputed = imputer.transform(X_predict)

# Train the RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_imputed, y_train)

# Predict the missing values
predicted_closeprice = model.predict(X_predict_imputed)

# Fill in the predicted values in the original dataframe
missing_groups.loc[:, 'median_closeprice'] = predicted_closeprice

# Merge the predicted missing groups back into the original dataframe
filled_filtered = pd.concat([filtered, missing_groups])

# Decode the encoded categorical columns back
for col in categorical_cols:
    le = label_encoders[col]
    filled_filtered.loc[:, col] = le.inverse_transform(filled_filtered[col].astype(int))

# Sort the DataFrame as per the specified columns
sorted_filled_filtered = filled_filtered.sort_values(by=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'])

# Calculate the year-over-year changes for median_closeprice
sorted_filled_filtered['median_closeprice_yoy_change'] = sorted_filled_filtered.groupby(['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN'], observed=True)['median_closeprice'].pct_change()

# Save the updated dataframe to a new CSV file
output_file_path_filled = 'C:\\cctaddr\\RES_SOLD_FILLED.csv'
sorted_filled_filtered.to_csv(output_file_path_filled, index=False)

print(f"Updated file saved to {output_file_path_filled}")

# Get the current year
current_year = datetime.now().year

# Function to calculate the updated close price
def calculate_updated_closeprice(row, yoy_changes, current_year):
    current_year_property = row['CLOSEYEAR']
    closeprice = row['CLOSEPRICE']
    # Apply year-over-year changes until the current year
    for year in range(current_year_property + 1, current_year + 1):
        yoy_change = yoy_changes.get(year, 0)
        closeprice *= (1 + yoy_change)
    return closeprice

# Create a dictionary to store year-over-year changes for each group
yoy_changes_dict = {}
for name, group in sorted_filled_filtered.groupby(['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN']):
    yoy_changes_dict[name] = group.set_index('CLOSEYEAR')['median_closeprice_yoy_change'].to_dict()

# Apply the function to calculate the updated close price
sold['UPDATED_CLOSEPRICE'] = sold.apply(
    lambda row: calculate_updated_closeprice(
        row,
        yoy_changes_dict.get((row['CITY'], row['PROPTYPE'], row['DIRECT'], row['TAX_BIN']), {}),
        current_year
    ),
    axis=1
)

# Merge the updated close prices back into the original dataframe
updated_sold = sold.copy()
updated_sold['CLOSEPRICE'] = updated_sold['UPDATED_CLOSEPRICE']
updated_sold.drop(columns=['UPDATED_CLOSEPRICE'], inplace=True)

# Save the updated dataframe to a new CSV file
output_file_path_currentized = 'C:\\cctaddr\\RES_SOLD_CURRENTIZED.csv'
updated_sold.to_csv(output_file_path_currentized, index=False)

print(f"Updated file saved to {output_file_path_currentized}")
