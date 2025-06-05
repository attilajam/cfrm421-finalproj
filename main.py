# main file where we import our models, and run them + evaluate their performance
import data_processing as dp
from models.random_forest import train_random_forest
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
   # load data
    df = dp.load_data()

    # preprocess data
    df = dp.preprocess_features(df)

    # split into train and test, using 0.001 of set as total dataset is 5mil
    train_df, test_df = dp.time_split(df, test_dates=3, sample_frac=0.001)

    print(train_df.shape)
    print(test_df.shape)

    # extract featuers and target
    X_train, y_train = dp.get_features_and_target(train_df)
    X_test, y_test = dp.get_features_and_target(test_df)

   # train lr
    train_lin_reg(X_train, y_train, X_test, y_test)

    # train rf
    train_random_forest(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
