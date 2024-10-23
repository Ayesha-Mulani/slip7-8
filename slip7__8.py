import numpy as np                      # For numerical operations
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.metrics import mean_absolute_error, r2_score  # For performance metrics

# Define the input data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # Reshape for sklearn
y = np.array([7, 14, 15, 18, 19, 21, 26, 23])          # Target variable

# Create and fit the linear regression model
model = LinearRegression()              # Instantiate the model
model.fit(x, y)                        # Train the model

# Get the estimated coefficients
b0 = model.intercept_                  # Intercept (b0)
b1 = model.coef_[0]                    # Slope (b1)

# Make predictions
y_pred = model.predict(x)              # Predict y values

# Evaluate the model's performance
mae = mean_absolute_error(y, y_pred)   # Calculate MAE
r2 = r2_score(y, y_pred)               # Calculate R² score

# Print the results
print(f'Estimated coefficient b0 (intercept): {b0}')  # Output b0
print(f'Estimated coefficient b1 (slope): {b1}')       # Output b1
print(f'Mean Absolute Error (MAE): {mae}')              # Output MAE
print(f'R-squared (R²): {r2}')                         # Output R²
