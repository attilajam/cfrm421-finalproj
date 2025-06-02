from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import data_processing as dp

def train_random_forest(df, test_dates=3, use_all=False):
    # preprocess features from dataframe
    df = dp.preprocess_features(df)
    
    # split data into training and test sets
    train_df, test_df = dp.time_split(df, test_dates=test_dates, use_all=use_all)
    
    # extract features and target from training set
    X_train, y_train = dp.get_features_and_target(train_df)
    
    if use_all:
        # If using all data, no test split - just train the model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        return rf
    
    # test set
    X_test, y_test = dp.get_features_and_target(test_df)
    
    # train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # predict on test set and evaluate performance
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R2: {r2:.4f}")
    
    return rf