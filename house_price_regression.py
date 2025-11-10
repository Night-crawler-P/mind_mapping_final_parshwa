import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic house pricing data
# Features: Size (sq ft), Bedrooms, Age (years)
np.random.seed(42)
n_samples = 200

# Generate features
size = np.random.randint(800, 3000, n_samples)  # House size in sq ft
bedrooms = np.random.randint(1, 5, n_samples)    # Number of bedrooms
age = np.random.randint(0, 30, n_samples)        # Age of house in years

# Generate target price (in thousands of dollars)
# Price = 50 * size + 20 * bedrooms - 2 * age + noise
price = 50 * size + 20 * bedrooms - 2 * age + np.random.normal(0, 30, n_samples)
price = np.maximum(price, 100)  # Ensure minimum price

# Create DataFrame
data = pd.DataFrame({
    'Size_sqft': size,
    'Bedrooms': bedrooms,
    'Age_years': age,
    'Price_thousands': price
})

print("House Pricing Dataset:")
print(data.head())
print("\nDataset Statistics:")
print(data.describe())

# Prepare features and target
X = data[['Size_sqft', 'Bedrooms', 'Age_years']]
y = data['Price_thousands']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)
print(f"\nTraining Set:")
print(f"  Mean Squared Error: {train_mse:.2f}")
print(f"  R² Score: {train_r2:.4f}")
print(f"\nTest Set:")
print(f"  Mean Squared Error: {test_mse:.2f}")
print(f"  R² Score: {test_r2:.4f}")

# Display model coefficients
print("\n" + "="*50)
print("MODEL COEFFICIENTS")
print("="*50)
print(f"Intercept: {model.intercept_:.2f}")
print(f"Size (sq ft) coefficient: {model.coef_[0]:.4f}")
print(f"Bedrooms coefficient: {model.coef_[1]:.4f}")
print(f"Age (years) coefficient: {model.coef_[2]:.4f}")

# Make some predictions on new data
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)
sample_houses = pd.DataFrame({
    'Size_sqft': [1500, 2000, 1200],
    'Bedrooms': [3, 4, 2],
    'Age_years': [5, 10, 15]
})

predictions = model.predict(sample_houses)
for i, (idx, house) in enumerate(sample_houses.iterrows()):
    print(f"\nHouse {i+1}:")
    print(f"  Size: {house['Size_sqft']} sq ft")
    print(f"  Bedrooms: {int(house['Bedrooms'])}")
    print(f"  Age: {int(house['Age_years'])} years")
    print(f"  Predicted Price: ${predictions[i]:.2f}K")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price (thousands $)')
axes[0, 0].set_ylabel('Predicted Price (thousands $)')
axes[0, 0].set_title('Actual vs Predicted Prices (Test Set)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Plot
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price (thousands $)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Price vs Size
axes[1, 0].scatter(data['Size_sqft'], data['Price_thousands'], alpha=0.5, label='Data')
# Plot regression line for size
size_range = np.linspace(data['Size_sqft'].min(), data['Size_sqft'].max(), 100)
bedrooms_avg = data['Bedrooms'].mean()
age_avg = data['Age_years'].mean()
price_line = model.coef_[0] * size_range + model.coef_[1] * bedrooms_avg + model.coef_[2] * age_avg + model.intercept_
axes[1, 0].plot(size_range, price_line, 'r-', lw=2, label='Regression Line')
axes[1, 0].set_xlabel('Size (sq ft)')
axes[1, 0].set_ylabel('Price (thousands $)')
axes[1, 0].set_title('Price vs House Size')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance (Coefficients)
features = ['Size (sq ft)', 'Bedrooms', 'Age (years)']
coefficients = model.coef_
axes[1, 1].barh(features, coefficients)
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Feature Importance (Coefficients)')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('house_price_regression_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*50)
print("Visualization saved as 'house_price_regression_results.png'")
print("="*50)

plt.show()

