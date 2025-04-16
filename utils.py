import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import ccf 



def ccf_plot(x, y, nlags=20, alpha=0.05):
    """
    Plot the Cross-Correlation Function (CCF) between two time series.

    Parameters:
    -----------
    x : array-like
        First time series.
    y : array-like
        Second time series.
    nlags : int
        Number of lags to compute.
    alpha : float
        Significance level for confidence intervals.

    Returns:
    --------
    Displays a CCF plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate CCF values
    ccf_vals = ccf(x, y, fft=True)[:nlags]  # extract only the values

    # Calculate standard error
    n = len(x)
    std_error = 1 / np.sqrt(n)

    # Calculate the z-score corresponding to the alpha value (for confidence interval)
    z_score = norm.ppf(1 - alpha / 2)  # Two-tailed test: 1 - alpha / 2

    # Calculate confidence interval
    confint_upper = std_error * z_score
    confint_lower = -std_error * z_score

    # Plot
    lags = np.arange(nlags)
    plt.figure(figsize=(10, 5))
    plt.bar(lags, ccf_vals, width=0.3, color='blue', edgecolor='black',label='CCF')

    # Plot confidence intervals
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=confint_upper, color='red', linestyle='dashed', label=f'{100*(1-alpha)}% CI')
    plt.axhline(y=confint_lower, color='red', linestyle='dashed')

    # Set fixed y-axis limits from -1 to 1
    plt.ylim(-1, 1)

    plt.xlabel("Lag")
    plt.ylabel("Cross-Correlation")
    plt.title("Cross-Correlation Function (CCF)")
    plt.xticks(lags)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def pccf(x, y, nlags=20):
    """
    Compute the Partial Cross-Correlation Function (PCCF) between two time series x and y.

    This function uses a recursive approach to calculate the partial cross-correlation 
    values up to the specified number of lags. It adjusts the raw cross-correlation 
    by removing the influence of intermediate lags, isolating the direct effect at each lag.

    Parameters:
    -----------
    x : array-like
        First time series (numpy array or list).
    y : array-like
        Second time series (must be same length as x).
    nlags : int, optional
        Number of lags for which to compute partial cross-correlation (default is 20).

    Returns:
    --------
    numpy.ndarray
        An array of partial cross-correlation values of length (nlags + 1).
        The first value is 1 (as PCCF(0) is always 1 by definition), followed by values for lag 1 to `nlags`.

    Notes:
    ------
    - This method is a simplified recursive approximation of the PACF logic applied to cross-correlation.
    - It assumes stationarity and linear dependence structure.

    Example:
    --------
    >>> pccf(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]), nlags=2)
    array([ 1.        , -1.        ,  0.        ])
    """
    acf_vals = ccf(x, y, nlags)
    pccf_vals = [acf_vals[0]]  # PCCF(0) = 1

    for k in range(1, nlags + 1):
        num = acf_vals[k]
        den = 1.0

        for j in range(1, k):
            num -= pccf_vals[j] * acf_vals[k - j]
            den -= pccf_vals[j] * acf_vals[j]

        pccf_k = num / den
        pccf_vals.append(pccf_k)

    return np.array(pccf_vals)


def pccf_plot(x, y, nlags=20, alpha=0.05):
    """
    Plot the Partial Cross-Correlation Function (PCCF) using the recursive method.

    Parameters:
    -----------
    x : array-like
        First time series.
    y : array-like
        Second time series.
    nlags : int
        Number of lags to compute PCCF.
    alpha : float
        Significance level (default 0.05 for 95% confidence interval).

    Returns:
    --------
    Displays a PCCF plot.
    """
    # Compute PCCF values using recursive algorithm
    pccf_vals = pccf(x, y, nlags=nlags)

    # Compute confidence interval bounds
    conf = norm.ppf(1 - alpha / 2) / np.sqrt(len(x))
    lower_bound = -conf
    upper_bound = conf
    lags = np.arange(nlags + 1)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(lags, pccf_vals, width=0.3, color='blue', edgecolor='black', label='PCCF')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(upper_bound, color='red', linestyle='--', label=f'{int((1-alpha)*100)}% CI')
    plt.axhline(lower_bound, color='red', linestyle='--')
    plt.ylim(-1, 1)

    plt.xlabel("Lag")
    plt.ylabel("Partial Cross-Correlation")
    plt.title("Partial Cross-Correlation Function (PCCF)")
    plt.xticks(lags)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def ecb_reshape(data, metric_name):
    """
    Reshape and clean ECB-style time series data.

    This function takes in a DataFrame containing time series data (e.g., from the European Central Bank),
    drops the second column (typically metadata or duplicate date info), renames the remaining columns for clarity,
    and returns a cleaned version of the data.

    Parameters:
    -----------
    data : pandas.DataFrame
        Original DataFrame with at least two columns: a date column and a metric column.
    metric_name : str
        Desired name for the metric column (e.g., 'Inflation', 'GDP', etc.).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with two columns: ['DATE', metric_name]. The 'DATE' column remains unchanged in format.

    Example:
    --------
    >>> ecb_reshape(df, 'Inflation')
        DATE    Inflation
    0   2022-01     5.1
    1   2022-02     5.9
    ...

    Notes:
    ------
    - The line to convert the DATE column to Year-Month format using `pd.to_datetime().dt.to_period("M")` is commented.
      Uncomment if needed for monthly time series standardization.
    """
    df = data.copy()
    df.drop(data.columns[1], axis=1, inplace=True)
    df.columns = ['DATE', metric_name]
    return df


def eurostat_reshape(data, metric_name, id_vars):
    """
    Reshape Eurostat-style wide-format data into a long, tidy format.

    This function is tailored for Eurostat datasets where multiple time columns are spread across the 
    columns and need to be melted into a single time series column. It then renames and cleans the 
    data for further use in analysis or visualization.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original wide-format Eurostat DataFrame, typically with geographic identifiers and many time columns.
    metric_name : str
        Desired name for the metric column after reshaping (e.g., 'Unemployment', 'Inflation').
    id_vars : list of str
        List of column names to keep as identifiers (e.g., ['geo\\TIME_PERIOD']).

    Returns:
    --------
    pandas.DataFrame
        A long-format DataFrame with three columns: ['DATE', 'GEO', metric_name].
        - 'DATE' contains the original time periods.
        - 'GEO' contains the geographical identifier.
        - metric_name contains the corresponding values.
    
    Notes:
    ------
    - Rows with missing values are removed.
    - The 'DATE' column is not yet converted to datetime. Uncomment the `pd.to_datetime()` line to convert if needed.
    - This function assumes the identifier column follows the Eurostat format like 'geo\\TIME_PERIOD'.

    Example:
    --------
    >>> eurostat_reshape(df, 'Unemployment', ['geo\\TIME_PERIOD'])
        DATE      GEO       Unemployment
    0   2021M01   DE        3.2
    1   2021M01   FR        4.8
    ...
    """
    data_melted = data.melt(id_vars=id_vars, 
                            var_name="TIME_PERIOD", 
                            value_name=metric_name)
    
    data_melted = data_melted[['TIME_PERIOD', "geo\\TIME_PERIOD", metric_name]]
    data_melted.columns = ['DATE', "GEO", metric_name]
    data_melted.dropna(inplace=True)
    
    # Convert to datetime format if needed
    # data_melted["DATE"] = pd.to_datetime(data_melted["DATE"]).dt.to_period("M")
    
    return data_melted


import requests
import json
import pandas as pd

def get_oecd_data(url):
    """
    Retrieve and process OECD data from a given JSON SDMX API URL.

    This function queries the OECD SDMX API, extracts observation data, maps 
    dimension codes to their respective values using structure metadata, and 
    returns a clean DataFrame.

    Parameters:
    -----------
    url : str
        The OECD SDMX API endpoint URL for the desired dataset.

    Returns:
    --------
    pandas.DataFrame
        A tidy DataFrame where each row is an observation, and columns represent
        all dimension values and the associated metric value under the 'Value' column.

    Raises:
    -------
    Exception:
        If the HTTP request fails or the response is not JSON-formatted.

    Example:
    --------
    >>> df = get_oecd_data("https://stats.oecd.org/SDMX-JSON/data/MEI_FIN/IR3TIB.GBR+USA.M/all")
    >>> df.head()
         FREQUENCY  LOCATION TIME_PERIOD  Value
    0          M        GBR      2022-01   0.25
    1          M        USA      2022-01   0.50
    ...

    Notes:
    ------
    - The function assumes SDMX-JSON format used by OECD's data API.
    - Each key in 'observations' is decoded by mapping index positions
      to dimension values using the structure metadata.
    """
    headers = {
        "Accept": "application/vnd.sdmx.data+json"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise Exception("Error: Response is not in JSON format.")

    # Extract dimensions and their metadata
    dimensions = data['data']['structures'][0]['dimensions']['observation']
    
    dimension_mapping = {}
    keyPosition_mapping = {}
    for dim in dimensions:
        dimension_mapping[dim['id']] = dim['values']
        keyPosition_mapping[dim['id']] = dim['keyPosition']

    # Extract observations and decode each using dimension positions
    observations = data['data']['dataSets'][0]['observations']
    rows = []
    for obs_key, obs_value in observations.items():
        indices = list(map(int, obs_key.split(':')))
        mapped_row = {}

        for dim_id, dim_values in dimension_mapping.items():
            position = keyPosition_mapping[dim_id]
            mapped_row[dim_id] = dim_values[indices[position]]['id']

        mapped_row["Value"] = obs_value[0]
        rows.append(mapped_row)

    return pd.DataFrame(rows)


def world_bank_reshape(data, metric_name, id_vars):
    """
    Reshape World Bank-style wide-format data into a tidy long-format DataFrame.

    This function converts time-based columns into a single 'DATE' column, 
    pairs each value with its corresponding country code, and renames columns 
    for easier analysis or visualization.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original wide-format World Bank dataset with country info and years as columns.
    metric_name : str
        Name to assign to the melted metric column (e.g., 'GDP', 'Population').
    id_vars : list of str
        List of identifier columns to keep during melt (e.g., ['Country Code']).

    Returns:
    --------
    pandas.DataFrame
        A long-format DataFrame with three columns: ['DATE', 'GEO', metric_name].
        - 'DATE': year or time period column (originally spread across many columns).
        - 'GEO': country code (or similar geographic identifier).
        - metric_name: actual observed value for that period and geography.

    Notes:
    ------
    - Removes rows with missing values.
    - The 'DATE' column is kept as string; uncomment the `pd.to_datetime()` line 
      to convert it to datetime format.
    - This function assumes 'Country Code' is the geographic identifier.

    Example:
    --------
    >>> world_bank_reshape(df, 'GDP', ['Country Code'])
        DATE     GEO     GDP
    0   2000     TUR    266.4
    1   2001     TUR    196.0
    ...
    """
    data_melted = data.melt(id_vars=id_vars, 
                            var_name="TIME_PERIOD", 
                            value_name=metric_name)

    data_melted = data_melted[['TIME_PERIOD', "Country Code", metric_name]]
    data_melted.columns = ['DATE', "GEO", metric_name]
    data_melted.dropna(inplace=True)

    # data_melted["DATE"] = pd.to_datetime(data_melted["DATE"]).dt.to_period("M")

    return data_melted
