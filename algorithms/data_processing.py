from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("./train.csv")
y = df["target"]
features = [col for col in df.columns if not col in ["target", "near_price", "far_price"]]
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
