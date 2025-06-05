import pandas as pd
import numpy as np

def load_data(path="./data/train.csv"):
    """
    Load CSV data from the given path, default is our train.csv file
    """
    return pd.read_csv(path)

def preprocess_features(df):
    """
    Feature engineering on raw dataframe.
    Add more features based on existing columns, to help
    machine learning models better capture patterns in the data. These features 
    include measures of market liquidity, imbalance pressure, and temporal dynamics, 
    such as bid-ask spread, price deviation from WAP, imbalance ratios, and 
    normalized time values.
    """
    # make copy of df to avoid modifying original
    df = df.copy()

    feature_cols = [
        "ask_price", "bid_price", "wap", "matched_size",
        "imbalance_size", "bid_size", "ask_size", "reference_price", 
        "seconds_in_bucket"
    ]
    # drop missing values
    df = df.dropna(subset=feature_cols)

    # new features and explanations
    # Bid-ask spread: measures market liquidity (the price gap buyers and sellers are willing to accept)
    df["spread"] = df["ask_price"] - df["bid_price"]

    # Midpoint price between best bid and ask, representing a fair estimate of current price
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2

    # Spread as a percentage of the weighted average price (wap), normalizing spread size
    df["spread_pct"] = df["spread"] / (df["wap"] + 1e-5)

    # Price impact: deviation of the reference price from the market average price (wap)
    df["price_impact"] = (df["reference_price"] - df["wap"]) / (df["wap"] + 1e-5)

    # Ratio of bid size to ask size, indicating buyer vs seller market pressure
    df["bid_ask_ratio"] = df["bid_size"] / (df["ask_size"] + 1e-5)

    # Market depth: total size available on both bid and ask sides (liquidity measure)
    df["market_depth"] = df["bid_size"] + df["ask_size"]

    # Imbalance ratio: relative size of auction imbalance compared to matched size, indicating auction pressure
    df["imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"] + 1e-5)

    # Absolute value of imbalance size to capture strength of imbalance regardless of direction
    df["abs_imbalance"] = df["imbalance_size"].abs()

    # Normalized time within the auction period (assuming 300 seconds total), to capture time effects
    df["time_norm"] = df["seconds_in_bucket"] / 300

    # Log-transformed matched size to reduce skewness and handle large volume variability
    df["log_matched_size"] = np.log1p(df["matched_size"])


    return df

def time_split(df, test_dates=3, sample_frac=None):
    """
    Split data by date_id as for stock prices, we need to
    ensure that time is respected in training

    test_dates are the number of most recent dates to reserve for testing, aka
    use up to the last N dates as test set

    Sample frac determines how much of the dataset we use, as the dataset is
    massive

    Returns a training set as a df and test set as a df
    """
    unique_dates = sorted(df["date_id"].unique())
    train_dates = unique_dates[:-test_dates]
    test_dates = unique_dates[-test_dates:]
    train_df = df[df["date_id"].isin(train_dates)]
    test_df = df[df["date_id"].isin(test_dates)]

    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)

    return train_df, test_df

def get_features_and_target(df, drop_id_cols=["stock_id", "date_id"]):
    """
    Split dataframe into features (X) and target (y)

    drop_id_cols determines whether to drop stock_id/date_id from features,
    as these are identifiers and not useful for prediction, merely seen as
    numbers by model which doesn't help it learn patterns. Defaults to
    dropping both stock_id and date_id, but can be customized to drop whatever

    Returns feature matrix as X and target vector as y
    """
    # drop target column as thats our predition target
    drop_cols = ["target"]
    # for each column in drop_id_cols, add it to the drop_cols list
    for col in drop_id_cols:
        drop_cols.append(col)
    # drop all specified columns from dataframe
    X = df.drop(columns=drop_cols)

    # target is the column we want to predict   
    y = df["target"]

    # return features and target
    return X, y