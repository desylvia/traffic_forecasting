import utils
import psutil
import cvxpy as cp
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


#------------------------------------------------------- Public vars
def p95(x):
    return x.quantile(.95)
def p70(x):
    return x.quantile(.70)
def p65(x):
    return x.quantile(.65)
def p60(x):
    return x.quantile(.60)

agg_mode = {
    'max': 'max',
    'mean': 'mean',
    'p95': p95,
    'p70': p70,
    'p65': p65,
    'p60': p65
}

num_horizon_by_granularity = {
    'D': 365,
    'SME': 24,
    'ME': 12
}

interval_by_granularity = {
    'D': 1,
    'SME': 14,
    'ME': 30
}

#------------------------------------------------------- Model constrained linear regression
class ConsLinReg:
    def __init__(self, freq, agg_mode, growth_factor):
        self.freq = freq
        self.agg_mode = agg_mode
        self.growth_factor = growth_factor
        
    def fit(self, df_train_, title, debug=False):
        self.ds_min = df_train_.ds.min()
        self.df_train_ = df_train_.copy()
        
        # variables for slope & intercept
        self.w_growth = cp.Variable() 
        self.intercept = cp.Variable()
        #self.w_naru = cp.Variable()
        
        self.constraints = []
        # slope ngga boleh negatif
        self.constraints.append(self.w_growth >= 0)
        #self.constraints.append(self.w_naru >= 0)
        
        self.w_last_ym_bias = cp.Variable()
            
        # get features
        idx = (df_train_["ds"] - self.ds_min).dt.days // interval_by_granularity[self.freq]
        self.df_train_["idx"] = idx

        last_ym = df_train_.yearmonth.max()
        last_ym_row = self.df_train_[df_train_.yearmonth == last_ym]
        last_ym_value = last_ym_row.agg({'y': self.agg_mode}).values[0]
        last_idx: int = last_ym_row.idx.values[0]
                
        # growth pada pada hasil forecasting di bulan terakhir tidak lebih besar dari
        # traffic di bulan terakhir data training (dikali dengan growth_factor)
        self.constraints.append(
                num_horizon_by_granularity[self.freq] * self.w_growth <= (last_ym_value * self.growth_factor)
        )
                
        # last_idx -> index terakhir dari data training
        # expr_last_ym_value -> traffic terakhir di data training
        expr_last_ym_value = (
                self.intercept
                + last_idx * self.w_growth
        )
        # memastikan traffic di bulan terakhir data training sesuai dengan hasil perhitungan
        # expr_last_ym_value + bias
        self.constraints.append(
                expr_last_ym_value + self.w_last_ym_bias == last_ym_value
        )

        # get actual traffic
        y = df_train_.y
        
        # get naru identifier
        #naru = df_train_.naru
            
        objective = cp.Minimize(cp.sum_squares(
                    self.intercept
                    + self.w_growth * idx
                    - y
            #objective = cp.Minimize(cp.sum_squares(
            #        self.intercept
            #        + self.w_growth * idx
            #        + self.w_naru * naru
            #        - y
        ))
        

        problem = cp.Problem(objective, self.constraints)
        problem.solve(solver=cp.SCS)
        
        status = problem.status
        
        if self.w_growth.value is None:  # previously "is not None"
            print(f"{title}. growth: {self.w_growth.value.round(2)}")
            
        if debug:
            print(f"solver status: {status}")
            print(title)
            print(f"intercept (titik awal): {self.intercept.value:.2f}")
            
        self.title = title
        self.ds_max = df_train_.ds.max()

        return self
    
    def predict(self, df_test_):
        dfnew = df_test_.copy()
        
        if self.freq == 'D':
            dfnew['idx'] = (dfnew['ds'] - self.ds_min).dt.days    
        elif self.freq == 'ME':
            dfnew["idx"] = (dfnew["ds"] - self.ds_min).dt.days // 30
        elif self.freq == 'SME':
            dfnew['idx'] = (dfnew['ds'] - self.ds_min).dt.days // 14
            
        # dfnew['idx'] = (dfnew['ds'] - self.ds_min).dt.days
        
        if self.w_growth.value is not None:
            self.w_growth.value = abs(self.w_growth.value)
            
        # print(f"slope: {self.w_growth.value}")
        dfnew['forecast'] = (
                self.intercept.value
                + self.w_growth.value * dfnew.idx
                + self.w_last_ym_bias.value
        )
        
        dfnew['title'] = self.title
        
        return dfnew
    
#------------------------------------------------------- Prediction function to call model
def predictions(
    ruas: str,
    df_train, df_test,
    granularity: str,
    agg_mode,
    growth_factor: float
):  
    cur_df_train = df_train.query("unique_id == @ruas")
    cur_df_test = df_test.query("unique_id == @ruas")
    
    errorlist = []
    
    if len(cur_df_train) <= 1:
        return

    try:
        model = ConsLinReg(granularity, agg_mode, growth_factor).fit(cur_df_train, ruas)
        pred = model.predict(cur_df_test)
        
        return pred
    
    except Exception as e:
        print("ERROR: ", ruas)
        print(model.constraints)
        errorlist.append(ruas)

#------------------------------------------------------- Seasonality computation
def calculate_deviation_in_percent(
        df_train, 
        ruas: str, 
        granularity: str, 
        agg_mode: str, 
        growth_factor: float,
        alpha: float=1.
    ) -> pd.DataFrame:
    """
    purpose:
        to calculate the deviation or difference between actual traffic and its trend 
        on each yearmonth
    """
    import model as model
    
    # get training data
    cur_df_train = df_train.query("unique_id == @ruas").copy()

    # train model
    conslinreg = model.ConsLinReg(
        granularity, 
        agg_mode, 
        growth_factor, 
    ).fit(cur_df_train, ruas)

    # get the trend
    pred = conslinreg.predict(cur_df_train.drop(columns=['y']))

    # calculate the deviation (in percent) between actual data and the trend (pred)
    df = cur_df_train.merge(
        pred.drop(columns=['idx','title','yearmonth']), 
        on=['unique_id','ds']
    ).rename(columns={
        'unique_id': 'ruas',
        'forecast': 'y_trend'
    })

    df['month_name'] = df.ds.dt.month_name()

    # calculate deviation
    df['deviation'] = df.apply(
        lambda x: utils.calculate_growth(x['y_trend'], x['y']), 
        axis=1
    )
    df['deviation'] = df['deviation'].map(
        lambda x: -1 if x<-1 else 1 if x>1 else x
    )
    
    # adjust the magnitude of deviation by percentage (alpha)
    df['deviation'] = df['deviation'] * alpha

    return df


def add_seasonality(df_deviation_agg, df_forecast):
    """
    purpose:
        to add the deviation effect from previous year
    """
    df_forecast_seasonal = df_forecast.copy()
    df_forecast_seasonal['month_name'] = pd.to_datetime(
        df_forecast_seasonal.yearmonth, 
        format="%Y-%m"
    ).dt.month_name()

    # merge each ruas with the particular deviation
    df_forecast_seasonal = df_forecast_seasonal.merge(
        df_deviation_agg, 
        on=['ruas','month_name'], 
        how='left'
    )
    
    # calculate the traffic with seasonality
    df_forecast_seasonal['forecast_seasonal'] = df_forecast_seasonal.apply(
        lambda x: (
            x['forecast'] + (x['forecast'] * x['avg_deviation'])
            if pd.notna(x['avg_deviation']) and pd.notna(x['forecast'])
            else x['forecast']
        ),
        axis=1
    )

    # calculate the traffic with seasonality
    #df_forecast_seasonal['forecast_seasonal'] = df_forecast_seasonal.apply(
    #    lambda x: x['forecast'] + (x['forecast']*x['avg_deviation']) \
    #        if pd.notna(x['forecast']) else x['forecast'],
    #    axis=1
    #)
    
    # fill nan-values
    df_forecast_seasonal['forecast_seasonal'] = df_forecast_seasonal.groupby(['ruas']).forecast_seasonal.bfill()
    df_forecast_seasonal['forecast_seasonal'] = df_forecast_seasonal.groupby(['ruas']).forecast_seasonal.ffill()

    # rename columns
    df_forecast_seasonal.drop(columns=['forecast', 'month_name', 'avg_deviation'], inplace=True)
    df_forecast_seasonal.rename(columns={'forecast_seasonal': 'forecast'}, inplace=True)

    columns_order = ['ruas', 'forecast', 'yearmonth']

    return df_forecast_seasonal[columns_order].copy()

#------------------------------------------------------- Forecasting: nits-simulation
def calculate_traffic_with_fixed_growth_rate(traffic, coeff, n_horizon):
    if n_horizon == N_HORIZON:
        global traffic_growth, initial_traffic
        
        traffic_growth = []
        initial_traffic = traffic

    elif n_horizon == 0:
        return traffic_growth
    
    # hitung growth
    growth = initial_traffic * (coeff/100)
    # tambahkan dengan current traffic
    traffic += growth
    traffic_growth.append(traffic)

    return calculate_traffic_with_fixed_growth_rate(traffic, coeff, n_horizon=n_horizon-1)


def forecast_nits(df, df_forecast, list_ruas, threshold: str, coeff: float):
    global N_HORIZON
    N_HORIZON = df_forecast.query("yearmonth >= @threshold").yearmonth.nunique()
    
    df_komparasi = df.query("ruas in (@list_ruas)").copy()
    df_komparasi['createddate'] = pd.to_datetime(df_komparasi['createddate'])
    df_komp_monthly = df_komparasi.groupby(
        ['ruas', pd.Grouper(key="createddate", freq='M')],
        as_index=False, 
        group_keys=True
    ).agg({'traffic': 'max'})
    df_komp_monthly['yearmonth'] = df_komp_monthly['createddate'].dt.to_period('M').astype(str)
    
    traffic_growth_all, list_ruas = [], []
    
    for ruas in df_komp_monthly.ruas.unique():
        try:
            traffic = df_komp_monthly.query(
                "yearmonth < @threshold and ruas == @ruas"
            ).iat[-1, -2]  # -2 means 'traffic' column index

            traffic_growth_all.append(
                calculate_traffic_with_fixed_growth_rate(traffic, coeff, N_HORIZON)
            )
            list_ruas.append([ruas for _ in range(N_HORIZON)])

        except Exception as e:
            print(f"{e} at ruas {ruas}")
            
    yearmonths = df_forecast.query(
        "yearmonth >= @threshold"
    ).yearmonth.unique().tolist()
    
    return yearmonths, pd.DataFrame({
        'ruas': np.array(list_ruas).flatten(),
        'yearmonth': np.array([yearmonths for _ in range(len(list_ruas))]).flatten(),
        'forecast': np.array(traffic_growth_all).flatten()
    })