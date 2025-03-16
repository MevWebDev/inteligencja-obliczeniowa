import pandas as pd

# Load the dataset and specify placeholders for missing values
df = pd.read_csv("./Lab2/iris_with_errors.csv", na_values=['-', 'nan', 'NaN', 'NAN'])

# Print the original dataset
print("Original dataset:")
print(df)



# Check for missing values
missing_values_count = df.isna().sum()
print("\nMissing values count per column:")
print(missing_values_count)

# Calculate the mean of only numeric columns and round to 1 decimal place
numeric_columns = df.select_dtypes(include=['number']).columns  # Get numeric columns
column_means = df[numeric_columns].mean().round(1)  # Calculate mean for numeric columns

# Fill missing values in numeric columns with their respective rounded means
df[numeric_columns] = df[numeric_columns].fillna(column_means)

# Check if 'species' column exists before filling missing values
if 'species' in df.columns:
    df['species'] = df['species'].fillna('unknown')
else:
    print("\nWarning: 'species' column not found in the dataset.")

# Define the valid range
valid_range = (0, 15)

# Replace out-of-range values with the column mean
for column in numeric_columns:
    out_of_range_mask = (df[column] < valid_range[0]) | (df[column] > valid_range[1])
    out_of_range_count = out_of_range_mask.sum()
    
    if out_of_range_count > 0:
        print(f"\nReplacing {out_of_range_count} out-of-range values in '{column}' with mean: {column_means[column]}")
        df.loc[out_of_range_mask, column] = column_means[column]

# Save the updated dataset
df.to_csv('iris_filled.csv', index=False)

# Load the filled dataset
df2 = pd.read_csv("./iris_filled.csv")

# Print the filled dataset
print("\nDataset after filling missing values and replacing out-of-range values:")
print(df2)

# Check for missing values in the filled dataset
missing_values_count_filled = df2.isna().sum()
print("\nMissing values count in filled dataset per column:")
print(missing_values_count_filled)

# Check for out-of-range values in the filled dataset
out_of_range_counts = {}

for column in numeric_columns:
    out_of_range = ((df2[column] < valid_range[0]) | (df2[column] > valid_range[1])).sum()
    out_of_range_counts[column] = out_of_range

# Print the results
print("\nNumber of values outside the range 0-15 per column after replacement:")
for column, count in out_of_range_counts.items():
    print(f"{column}: {count}")