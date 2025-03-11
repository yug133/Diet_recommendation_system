import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import gzip
import shutil
import time

# Import your functions here
from model import recommend, output_recommended_recipes, extract_ingredient_filtered_data

def is_gzip_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'

def optimize_extract_ingredient_filtered_data(dataframe, ingredients):
    if not ingredients:
        return dataframe
    
    # Convert ingredients to lowercase for case-insensitive matching
    ingredients_lower = [ing.lower() for ing in ingredients]
    
    # Use a vectorized operation instead of regex
    mask = dataframe['RecipeIngredientParts'].str.lower().apply(
        lambda x: all(ing in x for ing in ingredients_lower)
    )
    
    return dataframe[mask]

class TestRecipeRecommendation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = 'Data/dataset.csv'
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        
        # Check file size
        file_size = os.path.getsize(filepath)
        print(f"File size: {file_size} bytes")
        
        # Check if file is gzipped
        if is_gzip_file(filepath):
            print("File is gzipped. Attempting to decompress...")
            with gzip.open(filepath, 'rb') as f_in:
                with open('Data/dataset_decompressed.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath = 'Data/dataset_decompressed.csv'
        
        # Attempt to read the file
        try:
            cls.df = pd.read_csv(filepath, nrows=5)  # Try to read just the first 5 rows
            print("Successfully read the first 5 rows of the file.")
            print("Column names:", cls.df.columns.tolist())
            
            # Now read the entire file
            cls.df = pd.read_csv(filepath)
            print(f"Successfully read the entire file. Shape: {cls.df.shape}")
        except Exception as e:
            print(f"Error reading the file: {str(e)}")
            raise

        # Split the data into training and testing sets
        cls.train_df, cls.test_df = train_test_split(cls.df, test_size=0.2, random_state=42)

    def test_recommendation_accuracy(self):
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()

        # Use a smaller subset for testing to speed up the process
        test_subset = self.test_df.sample(n=min(1000, len(self.test_df)), random_state=42)

        for _, row in test_subset.iterrows():
            # Use the row's features as input
            input_features = row.iloc[6:15].values.reshape(1, -1)
            
            # Get recommendations
            recommended = recommend(self.train_df, input_features, extract_ingredient_filtered_data=optimize_extract_ingredient_filtered_data)
            
            if recommended is not None:
                recommended_output = output_recommended_recipes(recommended)
                
                # Check if the actual recipe is in the top 5 recommendations
                actual_name = row['Name']
                recommended_names = [recipe['Name'] for recipe in recommended_output]
                
                if actual_name in recommended_names:
                    correct_predictions += 1
                
                total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        end_time = time.time()
        print(f"Recommendation Accuracy: {accuracy:.2f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # You can set a threshold for acceptable accuracy
        self.assertGreater(accuracy, 0.5, "Model accuracy is below 50%")

    def test_ingredient_filtering(self):
        ingredients = ['chicken', 'rice']
        input_features = self.test_df.iloc[0, 6:15].values.reshape(1, -1)
        
        recommended = recommend(self.train_df, input_features, ingredients=ingredients, extract_ingredient_filtered_data=optimize_extract_ingredient_filtered_data)
        
        if recommended is not None:
            recommended_output = output_recommended_recipes(recommended)
            
            for recipe in recommended_output:
                recipe_ingredients = ' '.join(recipe['RecipeIngredientParts']).lower()
                self.assertTrue(all(ing in recipe_ingredients for ing in ingredients), 
                                f"Not all specified ingredients found in recipe: {recipe['Name']}")

if __name__ == '__main__':
    unittest.main()