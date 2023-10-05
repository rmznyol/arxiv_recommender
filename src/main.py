import pandas as pd 

df = pd.read_csv('data/minimal_surfaces_articles.csv')

print(df['Abstract'].iloc[1])