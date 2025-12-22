import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def info(df):
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nDataFrame Description:")
    print(df.describe())

if __name__ == "__main__":
    print("Loading and inspecting datasets...")
    userprofile_df = load_data('./userprofile.csv')
    info(userprofile_df)
    follows_df = load_data('./follows.csv')
    info(follows_df)
    posts_df = load_data('./posts.csv')
    info(posts_df)