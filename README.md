# House Price Linear Regression Model

A simple linear regression model to predict house prices based on features like size, number of bedrooms, and age. Includes both a standalone Python script and a Flask web application.

## Features

- **Size (sq ft)**: House size in square feet
- **Bedrooms**: Number of bedrooms
- **Age (years)**: Age of the house in years

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Web Application (Flask)

Run the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web interface allows you to:
- Enter house details (size, bedrooms, age)
- Get instant price predictions
- View model performance metrics
- See model coefficients and formula

### Standalone Script

Run the standalone Python script:

```bash
python house_price_regression.py
```

## What the Script Does

1. **Generates synthetic house pricing data** with realistic relationships
2. **Trains a linear regression model** using scikit-learn
3. **Evaluates the model** using Mean Squared Error (MSE) and R² score
4. **Makes predictions** on sample houses
5. **Visualizes results** with 4 plots:
   - Actual vs Predicted prices
   - Residuals plot
   - Price vs Size relationship
   - Feature importance (coefficients)

## Model Equation

The model predicts house price using:

```
Price = intercept + (coef_size × Size) + (coef_bedrooms × Bedrooms) + (coef_age × Age)
```

## Output

The standalone script will:
- Display dataset statistics
- Show model evaluation metrics
- Print model coefficients
- Make sample predictions
- Save visualization plots to `house_price_regression_results.png`

## Web Application Features

- **Interactive UI**: Modern, responsive web interface
- **Real-time Predictions**: Get instant price predictions
- **Model Information**: View model performance and coefficients
- **Formula Display**: See the exact prediction formula

