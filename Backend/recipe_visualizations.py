import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import scipy.stats as stats
import pylab
import gzip

# Step 1: Read the dataset
# Try reading as a gzip-compressed file (adjust if it's another compression format or needs different encoding)
with gzip.open('Data/dataset.csv', 'rt', encoding='utf-8') as f:
    data = pd.read_csv(f)

# Step 2: Display basic information
print(data.head())
print(data.info())

# Step 3: Plot a histogram of Calories
fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Calories Frequency Histogram')
plt.ylabel('Frequency')
plt.xlabel('Calories (Bin Centers)')
ax.hist(data.Calories.to_numpy(), bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 5000], linewidth=0.5, edgecolor="white")
plt.show()

# Step 4: Plot a probability plot for Calories
stats.probplot(data.Calories.to_numpy(), dist="norm", plot=pylab)
pylab.show()

# Step 5: Select necessary columns
columns = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeIngredientParts', 'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeInstructions']
dataset = data[columns]

# Step 6: Define maximum daily intake values
max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200

max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol, max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]

# Step 7: Filter the dataset based on maximum values
extracted_data = dataset.copy()
for column, maximum in zip(extracted_data.columns[6:15], max_list):
    extracted_data = extracted_data[extracted_data[column] < maximum]

# Step 8: Show the filtered data info and correlation matrix
print(extracted_data.info())
print(extracted_data.iloc[:, 6:15].corr())

# Step 9: Scale the data
scaler = StandardScaler()
prep_data = scaler.fit_transform(extracted_data.iloc[:, 6:15].to_numpy())

# Step 10: Nearest Neighbors model
neigh = NearestNeighbors(metric='cosine', algorithm='brute')
neigh.fit(prep_data)

# Step 11: Pipeline setup
transformer = FunctionTransformer(neigh.kneighbors, kw_args={'return_distance': False})
pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])

# Step 12: Update pipeline parameters
params = {'n_neighbors': 10, 'return_distance': False}
pipeline.set_params(NN__kw_args=params)

# Step 13: Test the pipeline with a sample input
test_input = extracted_data.iloc[0:1, 6:15].to_numpy()
recommended_indices = pipeline.transform(test_input)[0]
recommended_recipes = extracted_data.iloc[recommended_indices]
print(recommended_recipes)

# Step 14: Example filtering by ingredient (e.g., "egg")
filtered_data_by_ingredient = extracted_data[extracted_data['RecipeIngredientParts'].str.contains("egg", regex=False)]
print(filtered_data_by_ingredient)

# Step 15: Define the recommendation function
def recommend(dataframe, input_data, max_list, n_neighbors=5):
    extracted_data = dataframe.copy()
    
    # Filter data based on max daily intake
    for column, maximum in zip(extracted_data.columns[6:15], max_list):
        extracted_data = extracted_data[extracted_data[column] < maximum]
    
    # Scale the data
    prep_data = scaler.fit_transform(extracted_data.iloc[:, 6:15].to_numpy())
    
    # Fit the nearest neighbors model
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    
    # Build and use the pipeline to get the recommended recipes
    transformer = FunctionTransformer(neigh.kneighbors, kw_args={'return_distance': False})
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    pipeline.set_params(NN__kw_args={'n_neighbors': n_neighbors, 'return_distance': False})
    
    # Get the recommended recipes
    recommended_indices = pipeline.transform(input_data)[0]
    return extracted_data.iloc[recommended_indices]

# Step 16: Use the recommend function
recommended = recommend(dataset, test_input, max_list)
print(recommended)
