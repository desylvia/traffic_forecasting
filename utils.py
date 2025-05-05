import os
import psutil
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta


LAYER = "olt_me"
LAYER_DASH = "olt - metro"
MODEL_NAME = "ConsLinReg"

# -------------------------- Query definition
# 1) occupancy lama
sql_source_1 = """
SELECT node,
	   ip,
	   createddate,
	   reg,
	   metro,
	   sum(speed) as cap,
	   sum(traffic) as traffic,
	   'uplink' as mode
FROM (
    SELECT 
        node,
        ip,
        metro,
        speed,
        reg,
        createddate,
        (utilization / 100) * speed AS traffic,
        port,
        LOWER(SUBSTRING_INDEX(SUBSTRING_INDEX(port, ';', 1), ',', 1)) AS port_cleaned
    FROM occupancy_olt_uplink_daily
    WHERE utilization <= 100
) AS cleaned_data
where createddate < '2024-04-01'
GROUP BY node, createddate
ORDER BY node, createddate DESC"""

# 2) occupancy baru
# (round(((trf/100)/1.25/1000000),2)/(speed/1000000)*100) as utilization
# ((trf/100)/1.25)/1000000 AS traffic,
sql_source_2 = """
select node,
	   ip,
	   createddate,
	   reg,
	   max(metro) as metro,
	   sum(speed) as cap,
	   sum(usg) as traffic,
	   'uplink_n' as mode
from (
	select createddate,
	   	   reg,
	   	   node,
	   	   ip, 
	   	   metro, 
	   	   metro_port,
	   	   port,
	   	   LOWER(SUBSTRING_INDEX(SUBSTRING_INDEX(port, ';', 1), ',', 1)) AS port_cleaned,
	   	   (speed/1000000) as speed, 
	   	   round(((trf/100)/1.25/1000000),2) as usg
	from occupancy_olt_uplink_daily_n
	where (round(((trf/100)/1.25/1000000),2)/(speed/1000000)*100) <= 102
) AS cleaned_data
where createddate >= '2024-04-01'
GROUP BY node, createddate
ORDER BY node, createddate DESC"""

# 3) trunk_all_ruas
sql_source_3 = """
select tgl_file, sumber, ruas, nbr, no_name, nms_olt_ip
from trunk_all_ruas
where layer = 'olt - metro'
order by tgl_file desc"""

# --------------------------

def build_db_engine():
    host = "xxx"
    port = "xxx"
    username = "xxx"
    password = "xxx"
    database = "xxx"

    engine = create_engine(f"mysql://{username}:{password}@{host}:{port}/{database}")
    
    return engine


def pd_read_sql(query: str) -> pd.DataFrame:
    engine = build_db_engine()
    
    df = pd.read_sql(query, engine)
    
    return df

#================================================================== OLT FETCH DATA

def update_reg(x, node): # funtion to extract regional from node
    valid_reg_values = {'1', '2', '3', '4', '5', '6', '7'}
    
    # Check if x is a valid number between 1 and 7
    if pd.notna(x) and str(x).strip() != '' and str(x).replace('.', '', 1).isdigit():
        x_str = str(int(float(x)))
        if x_str in valid_reg_values:
            return x_str  # Return valid reg as a string
    
    # Otherwise, try to extract reg from 'node'
    parts = node.split('-')  # Split the 'node' value into parts
    
    if len(parts) > 1 and parts[1].startswith('D') and parts[1][1:].isdigit():
        return parts[1][1:]  # Extract the number after 'D'
    else:
        return '0'  # Return '0' if the pattern doesn't match

# function to check if old occ has already been fetched
def fetch_data():
    print("Fetching data from the source and saving it...")
    raw = pd_read_sql(sql_source_1)
    raw['node'] = raw['node'].str.strip()
    raw.to_csv("data/occ_old.csv", index=None)

def fetch_old_data(file_name: str, directory: str, fetch_function): 
    file_path = os.path.join(directory, file_name)
    
    if os.path.exists(file_path):
        print(f"'{file_name}' already exists in '{directory}'. Skipping fetch.")
        return file_path
    else:
        print(f"'{file_name}' not found. Fetching data...")
        fetch_data()  # Call the function to fetch data
        return file_path

def fetch_new_data(): # function to fecth new occ data
    raw = pd_read_sql(sql_source_2)
    raw['node'] = raw['node'].str.strip()
    raw.to_csv("data/occ_new.csv", index=None)
    
    occ_old = pd.read_csv("data/occ_old.csv")
    occ_old['cap'] = occ_old['cap'].astype('float64')
    
    occ = pd.concat([occ_old, raw], ignore_index=True)
    occ['reg'] = occ.apply(lambda row: update_reg(row['reg'], row['node']), axis=1)
    
    occ.to_csv("data/occ_all.csv", index=None)
    
    return "done"

# functions to look up with trunk_all_ruas
def get_ruas_and_sumber(sub_df): # Function to prioritize 'lldp' over 'descp'
    lldp_rows = sub_df[sub_df['sumber'] == 'lldp']
    descp_rows = sub_df[sub_df['sumber'] == 'descp']
    
    if not lldp_rows.empty:
        return pd.Series([lldp_rows.iloc[0]['ruas'], lldp_rows.iloc[0]['sumber']])
    elif not descp_rows.empty:
        return pd.Series([descp_rows.iloc[0]['ruas'], descp_rows.iloc[0]['sumber']])
    return pd.Series([None, None])

def fill_ruas(data): # main function to look up
    trunk = pd_read_sql(sql_source_3)
    trunk.tgl_file = pd.to_datetime(trunk.tgl_file)
    
    # Find the latest 'tgl_file' for each 'nbr'
    latest_dates = trunk.groupby('nbr')['tgl_file'].max().reset_index()
    latest_dates.columns = ['nbr', 'latest_date']

    # Merge back to filter trunk to latest date entries only
    filtered_trunk = pd.merge(trunk, latest_dates, left_on=['nbr', 'tgl_file'], right_on=['nbr', 'latest_date'])
    
    # Prioritize 'lldp' over 'descp'
    processed_trunk = filtered_trunk.groupby('nbr').apply(get_ruas_and_sumber, include_groups=False).reset_index()
    processed_trunk.columns = ['nbr', 'ruas', 'sumber']
    
    # Merge the processed 'trunk' with 'data'
    final_df = pd.merge(data, processed_trunk, left_on='node', right_on='nbr', how='left')
    final_df.drop(columns=['nbr'], inplace=True)  # Drop redundant 'nbr' column
    
    # Sort the result
    final_df = final_df.sort_values(by=['createddate', 'node'], ascending=[False, True]).reset_index(drop=True)
    df = final_df[['createddate', 'reg', 'node', 'metro', 'ruas', 'traffic', 'cap', 'sumber', 'mode']].copy()
    
    # Fill with node _to_ metro or node _to_ OTHER (when metro blank) if ruas is null
    # Replace NaN in 'metro' with 'OTHER'
    df['metro'] = df['metro'].fillna('OTHER')

    # Update 'ruas' only if it's NaN
    df['ruas'] = df.apply(lambda row: f"{row['node']}_to_{row['metro']}" if pd.isna(row['ruas']) else row['ruas'], axis=1)
    
    return df

#==================================================================

# Data modeling definition
def is_rafi(x):
    return x in ["2021-05", "2022-05", "2023-04", "2024-04"]

def create_statsforecast_df(df_):
    newdf = pd.DataFrame()
    #newdf["ruas"] = df_["ruas_alias"]
    newdf["unique_id"] = df_["ruas"]
    newdf["ds"] = pd.to_datetime(df_["createddate"])
    newdf["y"] = df_["traffic"]

    newdf["yearmonth"] = df_.yearmonth
    newdf["naru"] = df_.yearmonth.str[5:7] == "12"
    newdf["rafi"] = df_.yearmonth.apply(is_rafi)

    return newdf

def create_test_data(df_train, START_DATE, flag):
    # forecasting date range
    if (flag): # if flag = 1 then forecast long month
        if (datetime.now().strftime("%m") > '07'):
            END_DATE = datetime(datetime.now().year + 1, 12, 31).strftime("%Y-%m-%d")
        else:
            END_DATE = ((datetime.now() + relativedelta(months=12)) + relativedelta(day=31)).strftime("%Y-%m-%d")
    else: # if flag != 1 then forecast three months
        END_DATE = ((datetime.strptime(START_DATE, "%Y-%m-%d")) + relativedelta(months=2, day=31)).strftime("%Y-%m-%d")
    
    date_range = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="ME"
    )
    
    # define forecasting date range
    dfdate = pd.DataFrame({"ds": date_range, "key": 1})
    # define unique ruas
    unique_ids_df = pd.DataFrame({"unique_id": df_train.unique_id.unique(), "key": 1})

    df_test = (
        pd.merge(dfdate, unique_ids_df, on='key')
        .drop(columns='key')
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    df_test["yearmonth"] = df_test.ds.astype(str).str[:7]
    df_test["naru"] = df_test.yearmonth.str[5:7] == "12"
    
    return df_test

# Get old ruas (ruas with last traffic is > 4 months ago from now)
def extract_old_ruas(df):
    # Get the current year and month dynamically as a Period
    current_yearmonth = pd.Period(datetime.now(), freq='M')
    
    # Find the most recent 'yearmonth' for each 'ruas'
    latest_yearmonth = df.groupby('ruas')['yearmonth'].max().reset_index()

    # Convert the 'yearmonth' column to Period for the comparison without .dt
    latest_yearmonth['yearmonth'] = latest_yearmonth['yearmonth'].apply(lambda x: pd.Period(x, freq='M'))

    # Calculate the difference in months between the current yearmonth and the most recent yearmonth
    latest_yearmonth['month_diff'] = latest_yearmonth['yearmonth'].apply(lambda x: (current_yearmonth - x).n)
    
    # Filter to find ruas where the last traffic value is more than 4 months old
    ruas_with_old_traffic = latest_yearmonth[latest_yearmonth['month_diff'] > 4]
    
    return ruas_with_old_traffic

# Get new ruas (ruas with earliest entry is < 4 months ago from now)
def extract_new_ruas(df):
    # Get the current year and month dynamically as a Period
    current_yearmonth = pd.Period(datetime.now(), freq='M')
    
    # Find the oldest 'yearmonth' for each 'ruas'
    oldest_yearmonth = df.groupby('ruas')['yearmonth'].min().reset_index()
    
    # Convert the 'yearmonth' column to Period for the comparison
    oldest_yearmonth['yearmonth'] = oldest_yearmonth['yearmonth'].apply(lambda x: pd.Period(x, freq='M'))
    
    # Calculate the difference in months between the current yearmonth and the oldest yearmonth
    oldest_yearmonth['month_diff'] = oldest_yearmonth['yearmonth'].apply(lambda x: (current_yearmonth - x).n)
    
    # Filter to find ruas where the first entry is less than 4 months from the current month
    ruas_with_oldest_entries = oldest_yearmonth[oldest_yearmonth['month_diff'] < 4]
    
    return ruas_with_oldest_entries

# Get last value for old ruas
def last_value(df):
    current_yearmonth = pd.Period(datetime.now(), freq='M')
    
    # Initialize a list to collect new rows
    new_rows = []

    # Group the dataframe by 'ruas'
    for ruas, group in df.groupby('ruas'):
        # Sort by 'yearmonth' to ensure chronological order
        group = group.sort_values('yearmonth')
        
        # Get the last available row
        last_entry = group.iloc[-1]
        last_month = pd.Period(last_entry['yearmonth'], freq='M')
        
        # Create new rows for each missing month until the current month
        for month in pd.period_range(start=last_month + 1, end=current_yearmonth, freq='M'):
            new_row = last_entry.copy()
            new_row['yearmonth'] = month.strftime('%Y-%m')
            new_row['createddate'] = month.to_timestamp()  # Assuming createddate should also be updated
            new_row['traffic'] = np.nan  # Set traffic to NaN
            new_rows.append(new_row)
            
    # Convert new_rows to a DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Append new rows to the original DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
    # Forward fill the traffic column within each 'ruas'
    df['traffic'] = df.groupby('ruas')['traffic'].ffill()
    
    # Sort the DataFrame by 'ruas' and 'yearmonth' to maintain order
    df = df.sort_values(by=['ruas', 'yearmonth']).reset_index(drop=True)
    
    return df

# Define a function to handle outliers for multiple ruas
def handle_outliers_1(df, ruas_alias_col='ruas', date_col='createddate', traffic_col='traffic'): #Winsorization
    # Convert 'createddate' to datetime format
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')

    # Initialize an empty DataFrame to store the modified data
    df_after_winsorization = pd.DataFrame()

    # Get the unique 'ruas_alias' (road segments) to process each one separately
    ruas_list = df[ruas_alias_col].unique()

    for ruas in ruas_list:
        # Filter data for the current 'ruas_alias'
        df_latest_time_series = df[df[ruas_alias_col] == ruas].copy()

        # Calculate Z-scores for the 'traffic' column
        df_latest_time_series['z_score'] = stats.zscore(df_latest_time_series[traffic_col])

        # Mark the highest Z-score as the outlier
        max_z_score_idx = df_latest_time_series['z_score'].idxmax()
        df_latest_time_series['is_outlier'] = False  # Default to False
        df_latest_time_series.loc[max_z_score_idx, 'is_outlier'] = True

        # Sort data by date
        df_latest_time_series = df_latest_time_series.sort_values(by=date_col)

        # Identify consecutive outlier days
        df_latest_time_series['consecutive_group'] = (
            df_latest_time_series['is_outlier'].ne(df_latest_time_series['is_outlier'].shift())
            .cumsum()
        )

        # Group by 'consecutive_group' and calculate the duration of outliers in each group
        outlier_groups = df_latest_time_series[df_latest_time_series['is_outlier']].groupby('consecutive_group')[date_col].agg(['min', 'max', 'size'])

        # Apply Winsorization to outlier groups where the size is less than 5 consecutive days
        threshold = df_latest_time_series[traffic_col].quantile(0.99)
        for group_id in outlier_groups[outlier_groups['size'] < 5].index:
            df_latest_time_series.loc[df_latest_time_series['consecutive_group'] == group_id, traffic_col] = \
                df_latest_time_series.loc[df_latest_time_series['consecutive_group'] == group_id, traffic_col].apply(lambda x: min(x, threshold))

        # Explicitly cast 'is_outlier' to bool before concatenation
        df_latest_time_series['is_outlier'] = df_latest_time_series['is_outlier'].astype(bool)
        
        # Append the modified data for this 'ruas_alias' to the overall DataFrame
        df_after_winsorization = pd.concat([df_after_winsorization, df_latest_time_series], axis=0)

    return df_after_winsorization

def handle_outliers_2(df, ruas_alias_col='ruas', date_col='createddate', traffic_col='traffic'): #Second highest z_score
    # Convert 'createddate' to datetime format
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')

    # Initialize an empty DataFrame to store the modified data
    df_after_second_z = pd.DataFrame()

    # Get the unique 'ruas_alias' to process each one separately
    ruas_list = df[ruas_alias_col].unique()

    for ruas in ruas_list:
        # Filter data for the current 'ruas_alias'
        df_ruas = df[df[ruas_alias_col] == ruas].copy()

        # Calculate Z-scores for the 'traffic' column
        df_ruas['z_score'] = stats.zscore(df_ruas[traffic_col])

        # Identify the highest and second-highest Z-scores
        sorted_z_scores = df_ruas['z_score'].nlargest(2)
        if len(sorted_z_scores) > 1:
            max_z_score_idx = sorted_z_scores.index[0]  # Highest Z-score index
            second_max_traffic_value = df_ruas.loc[sorted_z_scores.index[1], traffic_col]  # Value of second-highest Z-score
            # Replace the traffic value corresponding to the highest Z-score with the second-highest value
            df_ruas.loc[max_z_score_idx, traffic_col] = second_max_traffic_value

        # Append the modified data for this 'ruas_alias' to the overall DataFrame
        df_after_second_z = pd.concat([df_after_second_z, df_ruas], axis=0)

    return df_after_second_z

def calculate_growth(t1, t2):
    try:
        # Use a ternary operator for simplicity
        return 0 if t1 == 0 else (t2 - t1) / t1
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#================================================================== EVALUATION

def mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    return np.mean(ape)


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def mape_adj(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    ape[~np.isfinite(ape)] = 0.0  # VERY questionable
    # ape[~np.isfinite(ape)] = 1. # pessimist estimate
    ape[ape >= 1] = 1
    return np.mean(ape)


def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y - y_bar) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    return 1 - (ss_res / ss_tot)


def rmse(y_actual, y_pred):
    return np.sqrt(np.mean((y_pred - y_actual) ** 2))


def mae(y_actual, y_pred):
    return np.mean(np.abs(y_actual - y_pred))


def ae(y_actual, y_pred):
    return np.abs(np.sum(y_actual) - np.sum(y_pred))


#def sape(y_actual, y_pred):
#    absolute_error = np.abs(y_actual - y_pred)
#    sape_ = absolute_error / (y_actual + y_pred)
#    return np.where(absolute_error <= 0.1, 0, sape_)

def sape(y_actual, y_pred):
    y_actual, y_pred = np.asarray(y_actual), np.asarray(y_pred)
    absolute_error = np.abs(y_actual - y_pred)
    denominator = y_actual + y_pred
    denominator = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
    sape_ = absolute_error / denominator
    sape_ = np.where(absolute_error <= 0.1, 0, sape_)
    return np.nan_to_num(sape_)

def smape(y_actual, y_pred):
    sape_ = sape(y_actual, y_pred)
    return np.mean(sape_)


def calculate_metrics(y_actual, y_pred):
    mape_ = mape(y_actual, y_pred)
    wmape_ = wmape(y_actual, y_pred)
    r2_ = r_squared(y_actual, y_pred)
    rmse_ = rmse(y_actual, y_pred)
    mae_ = mae(y_actual, y_pred)
    smape_ = smape(y_actual, y_pred)
    return [
        smape_,
        mape_,
        wmape_,
        r2_,
        rmse_,
        mae_,
    ]


def evaluation_model(y_actual, y_pred, layer, model, threshold, freq, train_period=None):
    """
    terdapat kondisi di mana tidak ada traffic baru di bulan x
    namun, muncul lagi di bulan x+1
    sehingga, data yang dievaluasi perlu disesuaikan
    dengan jumlah ruas yg available di data actual (done by merging)    
    """
    df_merge = pd.merge(
        y_actual.query("yearmonth == @threshold"),
        y_pred.query("yearmonth == @threshold"),
        how='inner',
        on=['ruas','yearmonth']
    )
    # excluding ruas dengan traffic=0 dan forecast=0
    df_merge = df_merge[df_merge.traffic + abs(df_merge.forecast) != 0].copy()
    
    # split data
    y_actual = df_merge['traffic']
    y_pred = df_merge['forecast']
    
    # evaluate forecasting result
    eval_result = calculate_metrics(y_actual, y_pred)
    columns = ["smape", "mape", "wmape", "r2_score", "rmse", "mae"]
    eval_result = pd.DataFrame(
        np.array(eval_result).reshape(1, len(columns)),
        columns=columns,
    )
    # add more info
    eval_result["layer"] = layer
    eval_result["model"] = model
    eval_result["yearmonth"] = threshold.replace('-','')
    
    if train_period is not None:
        # Do something if the variable is not None
        eval_result["yearmonth_train_set"] = (
            pd.to_datetime(train_period, format="%Y-%m") - relativedelta(months=1)
        ).strftime("%Y%m")
    else:
        # Handle the None case
        eval_result["yearmonth_train_set"] = (
            pd.to_datetime(threshold, format="%Y-%m") - relativedelta(months=1)
        ).strftime("%Y%m")
    
    eval_result["freq"] = freq
    eval_result["os_username"] = psutil.Process().username()
    eval_result["insert_dma"] = datetime.now()
    
    return eval_result

def get_eval_data(df, df_pred, threshold: str, agg_mode):
    # Convert daily data into monthly
    df_test = df.query(
        "yearmonth >= @threshold"
    ).groupby(['yearmonth', 'ruas'], as_index=False).agg({'traffic': agg_mode})
    
    print(f"[INFO] jumlah unique ruas data actual: {df_test.ruas.nunique()}")
    
    # Get needed forecasted traffic
    df_pred = df_pred.query("yearmonth >= @threshold")
    print(f"[INFO] jumlah unique ruas data prediction: {df_pred.ruas.nunique()}")
    
    return pd.merge(df_test, df_pred, on=['yearmonth', 'ruas'])
