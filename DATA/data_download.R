library("xts")

setwd("D://projects//DATA//BTCUSDT")

start_time = "2019-01-01 00:00:00"
btc_data = binance_klines('XRPUSDT', interval = '30m',start_time = start_time)
cdv = colnames(btc_data)
btc_X    = btc_data
xxc      = (btc_data[nrow(btc_data),1])[[1]]
start_time = xxc + 30*60

for(i in 1:60)
{
   print(i)
   btc_data = binance_klines('XRPUSDT', interval = '30m',start_time = start_time)
   colnames(btc_data) = cdv
   btc_X2   = btc_data
   xxc      = (btc_data[nrow(btc_data),1])[[1]]
   start_time = xxc + 30*60
   btc_X  = rbind(btc_X,btc_X2 )
   #saveRDS(btc_data ,file = paste0("btc_usdt",i,".rds"))
}

saveRDS(btc_X ,file = paste0("xrp_usdt",".rds"))
