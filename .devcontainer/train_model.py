import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Load dataset
df = pd.read_csv('cleaned_data.csv')

# Target variable
X = df.drop('Item_Outlet_Sales', axis=1)
y = np.log(df['Item_Outlet_Sales'])

# Categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, 'Ridge_Regression_best_model.pkl')

print("✅ Model trained and saved successfully!")
