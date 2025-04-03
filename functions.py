import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def ccf(x,y, length=20):
    return np.array([1]+[np.corrcoef(y[:-i].astype('float'), x[i:].astype('float'))[0,1]  \
        for i in range(1, length+1)])



def ccf_plot(x,y, nlags=20):
    """
    Plot the autocorrelation function (ACF) for a given time series.

    Parameters:
    - data: array-like, the time series data.
    - nlags: int, number of lags to compute ACF for.

    Returns:
    - Displays an ACF plot.
    """
    n = len(x)

    # Compute ACF values
    acf_values = ccf(x,y,length=nlags)

    # Confidence interval (95%) using 1.96 / sqrt(n)
    conf_int = 1.96 / np.sqrt(n)

    # Plot ACF
    plt.figure(figsize=(10, 5))
    plt.bar(range(nlags+1), acf_values, width=0.3, color='blue', label='ACF')
    plt.axhline(y=conf_int, color='red', linestyle='dashed', label='95% CI')
    plt.axhline(y=-conf_int, color='red', linestyle='dashed')
    plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.8)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function (ACF)")
    plt.legend()
    plt.show()


def pccf(x,y, nlags=20):
    acf_vals = ccf(x,y, nlags)
    pacf_vals = [1.0]  # PACF(0) = 1

    for k in range(1, nlags + 1):
        num = acf_vals[k]
        den = 1.0

        for j in range(1, k):
            num -= pacf_vals[j] * acf_vals[k - j]
            den -= pacf_vals[j] * acf_vals[j]

        pacf_k = num / den
        pacf_vals.append(pacf_k)

    return np.array(pacf_vals)



def pccf_plot(x,y, nlags=20, alpha=0.05):
    """
    Plot the Partial Autocorrelation Function (PACF) using the recursive method.

    Parameters:
    - x: array-like, the time series data.
    - nlags: int, number of lags to compute PACF.
    - alpha: float, significance level (default 0.05 for 95% confidence interval).
    """
    # Compute PACF values using recursive algorithm
    pacf_vals = pccf(x,y, nlags=nlags)

    # Compute confidence interval bounds
    conf = norm.ppf(1 - alpha / 2) / np.sqrt(len(x))
    lower_bound = -conf
    upper_bound = conf

    # Plot
    plt.figure(figsize=(10, 5))
    lags = np.arange(nlags + 1)

    plt.bar(lags, pacf_vals, width=0.3, color='blue', edgecolor='black', label='PACF')
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(upper_bound, color='red', linestyle='--', label=f'{int((1-alpha)*100)}% CI')
    plt.axhline(lower_bound, color='red', linestyle='--')
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.xticks(lags)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

