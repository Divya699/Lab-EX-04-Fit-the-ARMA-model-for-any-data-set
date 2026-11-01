
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 01/11/2026

### AIM:
To implement ARMA model in python.

# Dataset Used : canteen_shop_data
# Software Required:
- Jupyter Notebook
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


data = pd.read_csv("canteen_shop_data.csv")

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
X = data['Price']
print(data.head())

N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(data['Date'], X)
plt.title('Original Canteen Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(X, lags=min(len(X)//4, 35), ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(len(X)//4, 35), ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, min(500, N)])
plt.show()

plt.figure(figsize=(12, 5))
plot_acf(ARMA_1, ax=plt.gca())
plt.title('ACF - Simulated ARMA(1,1)')
plt.show()
plt.figure(figsize=(12, 5))
plot_pacf(ARMA_1, ax=plt.gca())
plt.title('PACF - Simulated ARMA(1,1)')
plt.show()

ar2 = np.array([
    1,
    -phi1_arma22,
    -phi2_arma22
])
ma2 = np.array([
    1,
    theta1_arma22,
    theta2_arma22
])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)
plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, min(500, N)])
plt.show()

plt.figure(figsize=(12, 5))
plot_acf(ARMA_2, ax=plt.gca())
plt.title('ACF - Simulated ARMA(2,2)')
plt.show()

plt.figure(figsize=(12, 5))
plot_pacf(ARMA_2, ax=plt.gca())
plt.title('PACF - Simulated ARMA(2,2)')
plt.show()
```
# OUTPUT:
# Original dataset

<img width="1097" height="592" alt="image" src="https://github.com/user-attachments/assets/0f9c3ebb-3e72-4dc2-8b5a-4158211e7ded" />

# Original Data ACF 

<img width="1101" height="356" alt="image" src="https://github.com/user-attachments/assets/9606ce05-b050-47c1-8e05-1384575a82cb" />

# Original Data PACF

<img width="919" height="305" alt="image" src="https://github.com/user-attachments/assets/71f02c40-c9b8-46fa-b13f-41c1a6beebb2" />

# SIMULATED ARMA(1,1) PROCESS:

<img width="888" height="476" alt="image" src="https://github.com/user-attachments/assets/f9754c96-de17-45ea-a890-d3b4fd81a25f" />


# Partial Autocorrelation

<img width="911" height="400" alt="image" src="https://github.com/user-attachments/assets/f4aa6e08-2f9e-46c2-a78f-0d567ddb41be" />


# Autocorrelation

<img width="915" height="393" alt="image" src="https://github.com/user-attachments/assets/2701ea00-1648-4221-8b86-decd67545af4" />


# SIMULATED ARMA(2,2) PROCESS:

<img width="910" height="471" alt="image" src="https://github.com/user-attachments/assets/c8eaff0d-ebcd-4712-9005-5744af814903" />


# Partial Autocorrelation

<img width="915" height="403" alt="image" src="https://github.com/user-attachments/assets/c65a8d17-9d4c-4075-a052-7bbed4d2d293" />


# Autocorrelation

<img width="915" height="397" alt="image" src="https://github.com/user-attachments/assets/ca85331a-5482-4c60-a790-9ad146ac2716" />


# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
