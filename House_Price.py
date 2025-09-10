import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Load data
    housing = pd.read_csv("C:/Users/talib/OneDrive/Desktop/Data science/project/Housing_Price/House__Price/main/housing.csv")

    # Create stratified test set based on median_income
    housing['income_cat'] = pd.cut(
        housing["median_income"], 
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("Input.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    # Separate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Identify numerical and categorical columns
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Build and fit pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Calculate RMSE on training data
    train_predictions = model.predict(housing_prepared)
    rmse = np.sqrt(mean_squared_error(housing_labels, train_predictions))
    print(f"Training RMSE: {rmse:.2f}")

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained and saved successfully!")

else:
    # Load model and pipeline for inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("Input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("Output.csv", index=False)
    print("Inference complete! Results saved to 0utput.csv")
