from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import data_processing as dp

def train_random_forest(X_train, y_train, X_test=None, y_test=None):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    if X_test is not None and y_test is not None:
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Test MSE: {mse:.6f}")
        print(f"Test R2: {r2:.4f}")
    
    return rf