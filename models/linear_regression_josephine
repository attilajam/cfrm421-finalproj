from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import data_processing as dp

def train_lin_reg(X_train, y_train, X_test=None, y_test=None):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    if X_test is not None and y_test is not None:
        y_pred = lr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Test MSE: {mse:.6f}")
        print(f"Test R2: {r2:.4f}")
    
    return lr
