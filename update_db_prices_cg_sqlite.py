# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:37:31 2025

@author: anima
"""

from pycoingecko import CoinGeckoAPI
import pandas as pd
import sqlite3

Ruta= "D:/Cripto/BasesSQLite"

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def generate_df_func(relation_func):

    cg= CoinGeckoAPI()
    
    data= cg.get_coin_market_chart_by_id(id=relation_func, vs_currency='usd', days=5)
    
    precios= data["prices"]
    
    df_1= pd.DataFrame(precios, columns=['timestamp', 'price'])
    
    volumenes= data['total_volumes']
    
    df_2= pd.DataFrame(volumenes, columns=['timestamp', 'volumen'])
    
    df_3= pd.merge(df_1, df_2, on='timestamp')
    
    df_3['timestamp'] = pd.to_datetime(df_3['timestamp'], unit='ms')
    
    df_4= df_3.rename(columns= {"price": "Close", "timestamp": "Datetime", "volumen": "Volume"})
    
    df_4['Day']= df_4['Datetime'].dt.day_name()

    cripto_history= df_4.copy()
    
    return cripto_history

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def delete_func(cripto_history_var, table_func):

    datetime_base= cripto_history_var["Datetime"][0]
    
    db= sqlite3.connect("{}/{}.db".format(Ruta, table_func))
    
    cur= db.cursor()
    
    delete_query = '''DELETE FROM {} WHERE Datetime >= '{}';'''.format(table_func, datetime_base)
    
    cur.execute(delete_query)
    
    db.commit()
    
    db.close()

    return 

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def update_table_func(cripto_history_var, table_func):

    db = sqlite3.connect("{}/{}.db".format(Ruta, table_func))
    
    cripto_history_var.to_sql('{}'.format(table_func), db, if_exists='append', index=False)
    
    db.close()

    return 

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

pairs = [
    ("bitcoin", "prices_bitcoin"),
    ("cardano", "prices_cardano"),
    ("ethereum", "prices_ethereum"),
    ("ripple", "prices_ripple"),
    ("solana", "prices_solana"),
]

for relation, table in pairs:

    cripto_history= generate_df_func(relation)
    
    delete_func(cripto_history, table)
    
    update_table_func(cripto_history, table)







#