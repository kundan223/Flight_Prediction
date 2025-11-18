import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Connect to SQLite database
engine = create_engine("sqlite:///data.db")

# Query the flight delay table into a DataFrame
df = pd.read_sql('SELECT * FROM table_name', engine)

# Store original column names for later
original_columns = df.columns.tolist()

# Convert 'Date' column to datetime and extract numeric features
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# Drop all datetime and timedelta columns to ensure no issues with XGBoost
df = df.select_dtypes(exclude=['datetime64', 'timedelta64'])

# Identify categorical columns (excluding target)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target column if it's in categorical columns
if 'LateAircraftDelay' in categorical_cols:
    categorical_cols.remove('LateAircraftDelay')

# Apply label encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Use astype(str) to handle potential mixed types before encoding
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoders for later use

# Prepare features and target
X = df.drop('LateAircraftDelay', axis=1)
y = df['LateAircraftDelay']

# Store feature names for later
feature_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Model RMSE: {rmse}')

# Save the trained model and encoders
joblib.dump(model, 'flight_delay_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# Load model and encoders
model = joblib.load('flight_delay_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

def preprocess_input(airline, origin, destination, date_str):
    """
    Preprocess user input to match training data format
    """
    # Create a dataframe with user input
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Origin': [origin],
        'Dest': [destination],
    })
    
    # Parse date and extract features
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    input_data['month'] = date_obj.month
    input_data['day'] = date_obj.day
    input_data['dayofweek'] = date_obj.weekday()
    
    # Apply label encoding
    for col in ['Airline', 'Origin', 'Dest']:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except:
                # Handle unknown categories
                input_data[col] = -1
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing columns with 0
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    return input_data

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            airline = request.form['airline']
            origin = request.form['origin']
            destination = request.form['destination']
            date = request.form['date']
            
            # Preprocess input data for prediction
            input_features = preprocess_input(airline, origin, destination, date)
            
            # Make prediction
            prediction = model.predict(input_features)
            prediction_result = round(prediction[0], 2)  # Round to 2 decimal places
            
        except Exception as e:
            error_message = str(e)  # Capture the error if any
    
    return render_template('index.html', 
                           prediction=prediction_result, 
                           error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
