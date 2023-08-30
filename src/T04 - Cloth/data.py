import pandas as pd

df = pd.read_csv('./final_test.csv')
df = df.dropna()
df = df.sample(frac=0.1)
df.to_csv('data_sampled.csv', index=False)