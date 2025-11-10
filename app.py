from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

app = Flask(__name__)

# Global variable to store the model
model = None
model_metrics = {}

def train_model():
    """Train the linear regression model"""
    global model, model_metrics
    
    # Generate synthetic house pricing data
    np.random.seed(42)
    n_samples = 200
    
    # Generate features
    size = np.random.randint(800, 3000, n_samples)
    bedrooms = np.random.randint(1, 5, n_samples)
    age = np.random.randint(0, 30, n_samples)
    
    # Generate target price (in thousands of dollars)
    price = 50 * size + 20 * bedrooms - 2 * age + np.random.normal(0, 30, n_samples)
    price = np.maximum(price, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Size_sqft': size,
        'Bedrooms': bedrooms,
        'Age_years': age,
        'Price_thousands': price
    })
    
    # Prepare features and target
    X = data[['Size_sqft', 'Bedrooms', 'Age_years']]
    y = data['Price_thousands']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Store metrics
    model_metrics = {
        'train_mse': round(train_mse, 2),
        'test_mse': round(test_mse, 2),
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
        'intercept': round(model.intercept_, 2),
        'size_coef': round(model.coef_[0], 4),
        'bedrooms_coef': round(model.coef_[1], 4),
        'age_coef': round(model.coef_[2], 4)
    }
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, model_metrics

def load_model():
    """Load the trained model if it exists, otherwise train a new one"""
    global model, model_metrics
    
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        # Recalculate metrics (or load from file)
        # For simplicity, we'll just train if model doesn't exist
        train_model()
    else:
        train_model()

# Initialize model on startup
load_model()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from request
        data = request.get_json()
        
        size = float(data.get('size', 0))
        bedrooms = int(data.get('bedrooms', 0))
        age = int(data.get('age', 0))
        
        # Validate inputs
        if size <= 0 or bedrooms <= 0 or age < 0:
            return jsonify({'error': 'Invalid input values. All values must be positive (age can be 0).'}), 400
        
        # Make prediction
        features = np.array([[size, bedrooms, age]])
        prediction = model.predict(features)[0]
        
        # Format prediction
        price_in_dollars = prediction * 1000  # Convert from thousands to dollars
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'price_dollars': round(price_in_dollars, 2),
            'features': {
                'size': size,
                'bedrooms': bedrooms,
                'age': age
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Return model information"""
    return jsonify(model_metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

