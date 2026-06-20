import pandas as pd

df = pd.read_csv("train.csv")
print(df.head())
pd.set_option("display.max_columns", None)
print(df.head())