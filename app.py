
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Connect to SQLite database
engine = create_engine("sqlite:///data.db")

# Query the flight delay table into a DataFrame
df = pd.read_sql('SELECT * FROM table_name', engine)

# Data Preprocessing
X = df.drop('target_column', axis=1)  # Assuming 'target_column' is the column you are predicting
y = df['target_column']  # Assuming this is the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the model (if needed for Flask)
import joblib
joblib.dump(model, 'flight_delay_model.pkl')
