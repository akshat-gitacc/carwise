import pandas as pd

df = pd.read_csv("../data/cardekho_dataset.csv")

df.drop(columns=['Unnamed: 0', 'car_name'], inplace=True)
df.to_csv("../data/processed_dataset.csv", index=False)