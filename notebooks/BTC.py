# %% [markdown]
### BTC analysis

# %%
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from scipy.signal import find_peaks


# %%
DATA_DIR = Path("data")

def fetch_historical_btc(symbol="btcusd"):
    # Bitstamp data starts around August 2011
    start_timestamp = int(datetime(2010, 8, 1).timestamp())
    end_timestamp = int(datetime.now().timestamp())
    
    url = f"https://www.bitstamp.net/api/v2/ohlc/{symbol}/"
    step = 86400  # 1 day in seconds
    limit = 1000  # Max limit per request allowed by Bitstamp
    
    all_data = []
    current_start = start_timestamp
    
    print(f"Fetching daily data for {symbol}...")
    
    while current_start < end_timestamp:
        params = {
            "step": step,
            "limit": limit,
            "start": current_start
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json().get("data", {}).get("ohlc", [])
        if not data:
            break
            
        all_data.extend(data)
        
        # Get the timestamp of the last retrieved candle and add one step
        last_timestamp = int(data[-1]["timestamp"])
        current_start = last_timestamp + step
        
        # Pause to respect the API rate limit (800 requests/minute)
        time.sleep(0.1)
        
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Clean up and format the DataFrame
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Select and convert necessary columns to float
    cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[cols].astype(float)
    
    # Drop any potential duplicates caused by overlapping API windows
    df = df[~df.index.duplicated(keep='first')]
    
    return df

btc_df = fetch_historical_btc()
print(btc_df.head())
print("\nTotal days fetched:", len(btc_df))
data_dir = Path.cwd().parent / "data"
    
# Create the directory if it does not exist
data_dir.mkdir(parents=True, exist_ok=True)

# Define the full file path and save the CSV
file_path = data_dir / "btc_daily_ohlcv.csv"
btc_df.to_csv(file_path)

print(f"Saved to {file_path}")

# %%
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def plot_btc_quantreg(df):
    # 1. Prepare data for the regression
    genesis_date = pd.to_datetime("2009-01-03")
    reg_df = df.copy()
    reg_df['days'] = (reg_df.index - genesis_date).days
    
    # Ensure no invalid log inputs
    reg_df = reg_df[(reg_df['days'] > 0) & (reg_df['close'] > 0)].copy()
    
    # Log-transform for the regression model
    reg_df['log_days'] = np.log10(reg_df['days'])
    reg_df['log_price'] = np.log10(reg_df['close'])
    
    # 2. Perform Quantile Regression
    quantiles = [0.10, 0.50, 0.90]
    predictions = {}
    
    # Fit log_price ~ log_days
    mod = smf.quantreg('log_price ~ log_days', reg_df)
    
    print("\nQuantile Regression Results (Power-Law Exponents):")
    for q in quantiles:
        res = mod.fit(q=q)
        print(f"q={q:.2f} | Beta (Slope): {res.params['log_days']:.4f} | Alpha (Intercept): {res.params['Intercept']:.4f}")
        
        # Transform predictions back to the linear scale for plotting on log-scaled axes
        predictions[q] = 10 ** res.predict(reg_df['log_days'])

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    halvings = [
        pd.to_datetime('2012-11-28'),
        pd.to_datetime('2016-07-09'),
        pd.to_datetime('2020-05-11'),
        pd.to_datetime('2024-04-20'),
    ]

    peaks = [
        pd.to_datetime('2013-12-04'), 
        pd.to_datetime('2017-12-16'), 
        pd.to_datetime('2021-11-08'), 
        pd.to_datetime('2025-10-06')
    ]

    bottoms = [
        pd.to_datetime('2015-01-14'), 
        pd.to_datetime('2018-12-15'), 
        pd.to_datetime('2022-11-21 '), 
        pd.to_datetime('2026-10-06')
    ]

    # --- Top Plot: Log-Log Scale with Quantile Regression Lines ---
    ax1.plot(reg_df.index, reg_df['close'], label='BTC Close', color='black', alpha=0.6, linewidth=1)
    
    colors = {0.10: 'green', 0.50: 'blue', 0.90: 'red'}
    styles = {0.10: '--', 0.50: '-.', 0.90: '--'}
    
    for q in quantiles:
        ax1.plot(reg_df.index, predictions[q], 
                 label=f'{int(q*100)}th Percentile Reg', 
                 color=colors[q], linestyle=styles[q], linewidth=2)

    for i, halving_date in enumerate(halvings):
        ax1.axvline(halving_date, color='purple', linestyle=':', alpha=0.7, linewidth=1.5,
                    label='Halving' if i == 0 else None)


    # Mark a notable BTC peak
    for i, peak_date in enumerate(peaks):
        ax1.axvline(peak_date, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                    label='Peak' if i == 0 else None)


    # Mark a notable BTC bottom
    for i, bottom_date in enumerate(bottoms):
        ax1.axvline(bottom_date, color='red', linestyle='-.', alpha=0.5, linewidth=1.5,
                    label='Bottom' if i == 0 else None)




    
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('BTC Price Dynamics - Power-Law Quantile Regression')
    ax1.set_xlabel('Days Since Genesis')
    ax1.set_ylabel('Price in USD')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)



    # --- Bottom Plot: Linear Scale ---
    # ax2.plot(reg_df.index, reg_df['close'], label='BTC Close', color='black')
    # ax2.set_title('BTC Price - Linear Scale')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('Price in USD')
    # ax2.legend()
    # ax2.grid(True, alpha=0.4)

    ax2.plot(reg_df.days, reg_df['close'], label='BTC Close', color='black', alpha=0.6, linewidth=1)
        
    colors = {0.10: 'green', 0.50: 'blue', 0.90: 'red'}
    styles = {0.10: '--', 0.50: '-.', 0.90: '--'}
    
    for q in quantiles:
        ax2.plot(reg_df.days, predictions[q], 
                    label=f'{int(q*100)}th Percentile Reg', 
                    color=colors[q], linestyle=styles[q], linewidth=2)
    

    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('BTC Price Dynamics - Power-Law Quantile Regression')
    ax2.set_xlabel('Days Since Genesis')
    ax2.set_ylabel('Price in USD')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


plot_btc_quantreg(btc_df)

# %%
def analyze_market_cycles(df):
    log_prices = np.log10(df['close'])
    
    peak_idx, _ = find_peaks(log_prices, distance=300, prominence=0.3)
    bottom_idx, _ = find_peaks(-log_prices, distance=300, prominence=0.3)
    
    peaks = df.iloc[peak_idx].copy()
    bottoms = df.iloc[bottom_idx].copy()
    
    mtgox_peak = pd.DataFrame({'close': [31.90]}, index=[pd.to_datetime('2011-06-08')])
    peaks = pd.concat([mtgox_peak, peaks]).sort_index()
    
    peaks['Days_Since_Last_Peak'] = peaks.index.to_series().diff().dt.days
    bottoms['Days_Since_Last_Bottom'] = bottoms.index.to_series().diff().dt.days
    
    print("=== HISTORICAL MACRO PEAKS ===")
    for date, row in peaks.iterrows():
        dist = f"{int(row['Days_Since_Last_Peak'])} days" if pd.notna(row['Days_Since_Last_Peak']) else "N/A"
        print(f"Date: {date.strftime('%Y-%m-%d')} | Price: ${row['close']:>9.2f} | Dist to prev: {dist}")
        
    peak_mean = peaks['Days_Since_Last_Peak'].mean()
    peak_std = peaks['Days_Since_Last_Peak'].std()
    print(f"\nPeak-to-Peak -> Mean: {peak_mean:.1f} days | Std Dev: {peak_std:.1f} days\n")
        
    print("=== HISTORICAL MACRO BOTTOMS ===")
    for date, row in bottoms.iterrows():
        dist = f"{int(row['Days_Since_Last_Bottom'])} days" if pd.notna(row['Days_Since_Last_Bottom']) else "N/A"
        print(f"Date: {date.strftime('%Y-%m-%d')} | Price: ${row['close']:>9.2f} | Dist to prev: {dist}")

    bottom_mean = bottoms['Days_Since_Last_Bottom'].mean()
    bottom_std = bottoms['Days_Since_Last_Bottom'].std()
    print(f"\nBottom-to-Bottom -> Mean: {bottom_mean:.1f} days | Std Dev: {bottom_std:.1f} days")

    return peaks, bottoms


peaks_df, bottoms_df = analyze_market_cycles(btc_df)
# %%
