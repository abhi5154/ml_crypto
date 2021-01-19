
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats


from ta import *
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.utils import dropna
from ta.volatility import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

def label_binary(dfx):
    labels1       = dfx.loc[:,'close'].shift(-1)/dfx.loc[:,'close'].shift(0) -1
    labels2       = np.where(labels1>0,1,-1)
    frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)

def label_multiple(dfx ,thresh):
    labels1       = dfx.loc[:,'close'].shift(-1)/dfx.loc[:,'close'].shift(0) -1
    labels2       = np.where(labels1>thresh,1,np.where(labels1< -thresh,-1,0))
    frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)

def load_all_feature_types(dfx):
    total_data_return   = dfx.shift(0)/dfx.shift(1) - 1
    technical_features  = technical_indicators(dfx)
    derived_features    = price_derivations(dfx)
    previous_features   = previous_returns(dfx)
    season_features     = seasonal_features(dfx)
    econ_features       = econometric_features(dfx)
    
    dfx2 = total_data_return

    dfx2 = dfx2.merge(technical_features,left_on='close_time', right_on='close_time')
    dfx2 = dfx2.merge(derived_features,left_on='close_time', right_on='close_time')
    dfx2 = dfx2.merge(previous_features,left_on='close_time', right_on='close_time')
    dfx2 = dfx2.merge(season_features,left_on='close_time', right_on='close_time')
    dfx2 = dfx2.merge(econ_features,left_on='close_time', right_on='close_time')

    
    return dfx2


def technical_indicators(df ):
    df_new       = pd.DataFrame()
    indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    indicator_ar = AroonIndicator(close=df["close"], window=10, fillna = False)
    
    df_new['sma5']   = df["close"]/sma_indicator(close=df["close"], window=5, fillna = False) -1
    df_new['sma10']  = df["close"]/sma_indicator(close=df["close"], window=10, fillna = False) -1
    df_new['sma20']  = df["close"]/sma_indicator(close=df["close"], window=20, fillna = False) -1
    
    df_new['ema5']   = df["close"]/ema_indicator(close=df["close"], window=5, fillna = False) -1
    df_new['ema10']  = df["close"]/ema_indicator(close=df["close"], window=10, fillna = False) -1
    df_new['ema20']  = df["close"]/ema_indicator(close=df["close"], window=20, fillna = False) -1

    df_new['rsi14']  = (df["close"]/rsi(close=df["close"], window=14, fillna = False) -50)/100
    df_new['rsi28']  = (df["close"]/rsi(close=df["close"], window=28, fillna = False) -50)/100

    #df_new['cmo28']  = CMO(gbpinr ,n = 14)
    df_new['aro10']  = AroonIndicator(close=df["close"], window=10, fillna = False).aroon_indicator()
    df_new['aro20']  = AroonIndicator(close=df["close"], window=20, fillna = False).aroon_indicator()
    
    df_new['bb_bbm'] = df["close"]/indicator_bb.bollinger_mavg()  -1
    df_new['bb_bbh'] = df["close"]/indicator_bb.bollinger_hband() -1
    df_new['bb_bbl'] = df["close"]/indicator_bb.bollinger_lband() -1
    
    df_new['cci10']   = CCIIndicator(high = df["high"],low = df["low"],close = df["close"] , window= 10 , fillna = False).cci()
    df_new['cci20']   = CCIIndicator(high = df["high"],low = df["low"],close = df["close"] , window= 20 , fillna = False).cci()
    #df_new['cci40']
    
    df_new['chk10']  = ChaikinMoneyFlowIndicator(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"] , window= 10 , fillna = False).chaikin_money_flow()
    df_new['chk20']  = ChaikinMoneyFlowIndicator(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"] , window= 20 , fillna = False).chaikin_money_flow()
    
    df_new['macd']   = MACD(df["close"],fillna = False).macd()
    df_new['stk']    = StochasticOscillator(high = df["high"],low = df["low"],close = df["close"],window= 10 , fillna = False).stoch()
    
    df_new['adi']    = AccDistIndexIndicator(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"]  , fillna = False).acc_dist_index()
    df_new['nvi']    = NegativeVolumeIndexIndicator(close = df["close"],volume = df["volume"] , fillna = False).negative_volume_index()
    df_new['obv']    = OnBalanceVolumeIndicator(close = df["close"],volume = df["volume"] , fillna = False).on_balance_volume()

    df_new['emi10']    = EaseOfMovementIndicator(high = df["high"],low = df["low"],volume = df["volume"] , window= 10 , fillna = False).ease_of_movement()
    df_new['emi20']    = EaseOfMovementIndicator(high = df["high"],low = df["low"],volume = df["volume"] , window= 20 , fillna = False).ease_of_movement()
    
    df_new['vwap5']    = df["close"]/VolumeWeightedAveragePrice(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"],window = 5  , fillna = False).volume_weighted_average_price() -1
    df_new['vwap10']   = df["close"]/VolumeWeightedAveragePrice(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"],window = 10  , fillna = False).volume_weighted_average_price() -1
    df_new['vwap20']   = df["close"]/VolumeWeightedAveragePrice(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"],window = 20  , fillna = False).volume_weighted_average_price() -1
    df_new['vwap40']   = df["close"]/VolumeWeightedAveragePrice(high = df["high"],low = df["low"],close = df["close"],volume = df["volume"],window = 40  , fillna = False).volume_weighted_average_price() -1


    
    return df_new

def reversion(x):
    x1    = (x - x.shift(1)).values

    x2    = x.shift(1).values
    mask = ~np.isnan(x1) & ~np.isnan(x2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x2[mask], x1[mask])
    return slope

def autocorr(x):
    x1    = (x/x.shift(1)).values - 1
    x2    = (x.shift(1)/x.shift(2)).values -1
    mask = ~np.isnan(x1) & ~np.isnan(x2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x2[mask], x1[mask])
    return slope

def retlag(x):
    x1    = (x/x.shift(1)).values -1
    x2    = (x.shift(1)/ x.shift(2)).values -1
    x3    = (x.shift(2)/x.shift(7)).values -1
    mask = ~np.isnan(x1) & ~np.isnan(x2)& ~np.isnan(x3)
    dfx =   pd.DataFrame()
    dfx['x2'] = x2[mask]
    dfx['x3'] = x3[mask]
    
    model = LinearRegression()
    model.fit(dfx ,x1[mask])

    return model.coef_[1]


def price_derivations(df):
    
    df_new        = pd.DataFrame(index = df.index)
    return_series = df["close"]/df["close"].shift(1) -1
    return_volume = df["volume"]/df["volume"].shift(1) -1
    close         = df["close"]
    raw_volume    = df["volume"]
    raw_trades    = df["trades"]
    
    rets          = return_series
    volrets       = np.sign(rets)*np.sign(return_volume)
    
    sign   = np.sign(rets)
    of1    = sign*raw_volume
    of2    = sign*np.log(raw_volume)

    df_new["raw_volume"] = raw_volume
    df_new["raw_trades"] = raw_trades
    df_new["reversion"]  = close.rolling(20).apply(reversion)
    
    df_new["ret5mean"]  = rets.rolling(5).mean()
    df_new["ret10mean"] = rets.rolling(10).mean()
    df_new["ret20mean"] = rets.rolling(20).mean()
    
    df_new["volret5mean"]  = volrets.rolling(5).mean()
    df_new["volret10mean"] = volrets.rolling(10).mean()
    df_new["volret20mean"] = volrets.rolling(20).mean()


    df_new["ofa5"]  = of1.rolling(5).mean()
    df_new["ofb5"]  = of2.rolling(5).mean()
    df_new["ofa10"] = of1.rolling(10).mean()
    df_new["ofb10"] = of2.rolling(10).mean()
    df_new["ofa20"] = of1.rolling(20).mean()
    df_new["ofb20"] = of2.rolling(20).mean()
    df_new["ofa40"] = of1.rolling(40).mean()
    df_new["ofb40"] = of2.rolling(40).mean()

    df_new["skew10"] = return_series.rolling(10).skew()
    df_new["skew20"] = return_series.rolling(20).skew()
    df_new["skew40"] = return_series.rolling(40).skew()
    
    df_new["kurt10"] = return_series.rolling(10).kurt()
    df_new["kurt20"] = return_series.rolling(20).kurt()
    df_new["kurt40"] = return_series.rolling(40).kurt()
    
    
    # df_new["acf20"]  = close.shift(1).rolling(20).apply(autocorr)
    # df_new["acf40"]  = close.shift(1).rolling(40).apply(autocorr)
    
    # df_new["retlag20"]  = close.rolling(20).apply(retlag)
    # df_new["retlag40"]  = close.rolling(40).apply(retlag)


    
    return df_new
    
def previous_returns(df):
    
    df_new            = pd.DataFrame()
    return_series     = df["close"]/df["close"].shift(1) -1
    close             = df["close"]
    df_new["retl11"]  = df["close"].shift(1)/df["close"].shift(2) -1
    df_new["retl12"]  = df["close"].shift(2)/df["close"].shift(3) -1
    df_new["retl50"]  = df["close"].shift(0)/df["close"].shift(5) -1
    df_new["retl51"]  = df["close"].shift(1)/df["close"].shift(6) -1
    df_new["retl100"] = df["close"].shift(0)/df["close"].shift(10) -1
    df_new["retl101"] = df["close"].shift(1)/df["close"].shift(11) -1
    df_new["retl200"] = df["close"].shift(0)/df["close"].shift(20) -1
    df_new["retl201"] = df["close"].shift(1)/df["close"].shift(21) -1
    
    

    return df_new

def econometric_features(df):
    
    df_new        = pd.DataFrame(index = df.index)
    return_series = df["close"]/df["close"].shift(1) -1
    close         = df["close"]
    raw_volume    = df["volume"]
    raw_trades    = df["trades"]
    volat_series  = return_series*return_series
    
    bp0   = np.abs(return_series)*np.abs(return_series.shift(1))
    sd0   = return_series*return_series
    gc0   = np.sqrt(volat_series.ewm(span = 2.0/0.9-1).mean())
    
    df_new["gc0"]      = gc0
    
    df_new["std5"]   = return_series.rolling(5).std()
    df_new["std10"]  = return_series.rolling(10).std()
    df_new["std20"]  = return_series.rolling(20).std()
    df_new["std40"]  = return_series.rolling(40).std()

    
    df_new["bpv5"]  = bp0.rolling(5).mean()
    df_new["bpv10"] = bp0.rolling(10).mean()
    df_new["bpv20"] = bp0.rolling(20).mean()
        
    df_new["bpvstd5"]  = df_new["bpv5"] - df_new["std5"]
    df_new["bpvstd10"] = df_new["bpv10"] - df_new["std10"]
    df_new["bpvstd20"] = df_new["bpv20"] - df_new["std20"]

    df_new["gc0bp5"]   = df_new["gc0"] - df_new["bpv5"]
    df_new["gc0bp10"]  = df_new["gc0"] - df_new["bpv10"]
    df_new["gc0bp20"]  = df_new["gc0"] - df_new["bpv20"]
    
    df_new["gc0std5"]   = df_new["gc0"] - df_new["std5"]
    df_new["gc0std10"]  = df_new["gc0"] - df_new["std10"]
    df_new["gc0std20"]  = df_new["gc0"] - df_new["std20"]
    

    return df_new
    
def seasonal_features(df):
    df_new        = pd.DataFrame(index = df.index)
    return_series = df["close"]/df["close"].shift(1) -1
    close         = df["close"]
    raw_volume    = df["volume"]
    raw_trades    = df["trades"]
    rets   = return_series
    sign   = np.sign(rets)
    of1    = sign*raw_volume
    of2    = sign*np.log(raw_volume)

    
    label_encoder = LabelEncoder()
    ttr           = df.index
    ttr           = ttr.strftime("%H:%M").values
    sdx           = label_encoder.fit_transform(ttr)
    
    df_new["time_label"]= sdx
    df_new["rets1"]      = rets.shift(48)
    df_new["rets0"]      = rets.shift(24)
    
    vol_s1   = (raw_volume.shift(1*48) + raw_volume.shift(2*48) + raw_volume.shift(3*48) + raw_volume.shift(4*48) + raw_volume.shift(5*48))/5
    trd_s1   = (raw_trades.shift(1*48) + raw_trades.shift(2*48) + raw_trades.shift(3*48) + raw_trades.shift(4*48) + raw_trades.shift(5*48))/5
    
    volumes1 = raw_volume/vol_s1 - 1
    trades1  = raw_trades/trd_s1 - 1


    df_new["volumes1"] = volumes1
    df_new["trades1"]  = trades1
    df_new["volumei1"] = np.where(volumes1 > 0.5,1,0)*rets
    df_new["tradesi1"] = np.where(trades1 > 0.5,1,0)*rets
    df_new["volumei2"] = np.where(volumes1 > 1,1,0)*rets
    df_new["tradesi2"] = np.where(trades1 > 1,1,0)*rets
    df_new["volumei3"] = np.where(volumes1 > 2,1,0)*rets
    df_new["tradesi3"] = np.where(trades1 > 2,1,0)*rets


    
    
    return df_new

def factor_features(df):
    return 0
    







