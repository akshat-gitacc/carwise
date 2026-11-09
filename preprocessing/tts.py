import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed_dataset.csv")

x = df.drop(columns=['selling_price'])
y = df['selling_price']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)