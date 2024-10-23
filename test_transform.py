# tests/test_transform.py

import pytest
import pandas as pd
from data_engineering.transform import transform_data

def test_transform_data_normalization():
    # Sample data
    data = {
        "customer_id": [1, 2, 3],
        "age": [25, 35, 45],
        "balance": [1000.0, 2000.0, 3000.0]
    }
    df = pd.DataFrame(data)

    # Call the transform_data function
    transformed_df = transform_data(df)

    # Assertions
    assert "age" in transformed_df.columns, "age column should exist after transformation"
    assert "balance" in transformed_df.columns, "balance column should exist after transformation"

    # Check normalization: age and balance should be between 0 and 1
    assert transformed_df["age"].min() == 0.0, "age column not normalized correctly"
    assert transformed_df["age"].max() == 1.0, "age column not normalized correctly"
    assert transformed_df["balance"].min() == 0.0, "balance column not normalized correctly"
    assert transformed_df["balance"].max() == 1.0, "balance column not normalized correctly"

def test_transform_data_handle_missing_values():
    # Sample data with missing values
    data = {
        "customer_id": [1, 2, 3],
        "age": [25, None, 45],
        "balance": [1000.0, 2000.0, None]
    }
    df = pd.DataFrame(data)

    # Call the transform_data function
    transformed_df = transform_data(df)

    # Assertions
    assert len(transformed_df) == 1, "Rows with missing values should be dropped"
    assert not transformed_df.isnull().values.any(), "There should be no missing values after transformation"
