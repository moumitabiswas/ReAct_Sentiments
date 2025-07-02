import pandas as pd
import os
code_path = r""
# Change to the directory where the code is located
os.chdir(code_path) 

# path=r"C:\Users\mbiswas\OneDrive - Capgemini\Documents\MBiswas2025\Custom_Framework_Prompt\ReAct_Sentiments\data_load\Review.csv"
# Load the dataset for review data
def dataload(path):
    df_rev= pd.read_csv(path)
    df_rev= df_rev.sample(2000, random_state=42)  # Sample 1000 rows for testing
    df_rev.reset_index(drop=True, inplace=True)
    df_rev=df_rev[df_rev['Language'] == 'en']  # Filter for English reviews
    return df_rev

def inputtext(df):
    text=df['review_clean'].to_list()
    return text


