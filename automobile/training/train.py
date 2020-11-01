import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Prepare data
def prepare_data(data):
    data["horsepower"] = pd.to_numeric(data["horsepower"], errors='coerce')
    data=data.fillna(data.mean())
    data=data.drop(['car name'],axis=1)
    return data

# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('mpg', axis=1).values
    y = df['mpg'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data
# Train the model, return the model
def train_model(data, args):
    reg_model = LinearRegression(normalize=True)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

# Evaluate the metrics for the model
def get_model_metrics(reg_model, data):
    preds = reg_model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics


def main():
    # Load Data
    sample_data = pd.read_csv('auto-mpg.csv')

    df = pd.DataFrame(
        data=sample_data.data,
        columns=sample_data.feature_names)
    df['mpg'] = sample_data.target
    df=prepare_data(df)
    # Split Data into Training and Validation Sets
    data = split_data(df)
    # Train Model on Training Set
    args = { }
    reg = train_model(data, args)

    # Validate Model on Validation Set
    metrics = get_model_metrics(reg, data)

    # Save Model
    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)

if __name__ == '__main__':
    main()
