import pandas as pd
#import matplotlib.pyplot as plt
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the CSV file
csv_file_path = os.path.join(script_dir, 'data', 'tutorial_round.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Work with the DataFrame
display(df.head())

amethysts_df = df[df['product'] == 'AMETHYSTS']
