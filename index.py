python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the SAIFI data
data = pd.read_csv('SAIFI.csv')

# Display the first few rows of the dataset
print(data.head())

# Extract the year and SAIFI values
years = data['Year'].values.reshape(-1, 1)
saifi_values = data['SAIFI'].values

# Plot the historical SAIFI data
plt.figure(figsize=(10, 6))
plt.plot(years, saifi_values, marker='o', label='Historical SAIFI')
plt.title('Annual SAIFI Values (2013-2025)')
plt.xlabel('Year')
plt.ylabel('SAIFI Value')
plt.grid(True)
plt.legend()
plt.show()

# Perform predictive analysis using linear regression
model = LinearRegression()
model.fit(years, saifi_values)

# Predict future values
future_years = np.arange(2026, 2031).reshape(-1, 1)
future_saifi = model.predict(future_years)

# Combine historical and predicted values
all_years = np.concatenate((years, future_years), axis=0)
all_saifi = np.concatenate((saifi_values, future_saifi), axis=0)

# Plot historical and predicted SAIFI values
plt.figure(figsize=(10, 6))
plt.plot(all_years, all_saifi, marker='o', label='SAIFI Trend')
plt.axvline(x=2025, color='red', linestyle='--', label='Prediction Start')
plt.title('Historical and Predicted SAIFI Values')
plt.xlabel('Year')
plt.ylabel('SAIFI Value')
plt.grid(True)
plt.legend()
plt.show()
