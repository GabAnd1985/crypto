# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:37:31 2025

@author: anima
"""

from pycoingecko import CoinGeckoAPI
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def keys_func():

    Ruta= "D:/Cripto"
    
    fileref= open("{}/keys.txt".format(Ruta), "r")
    
    lines= fileref.readlines()
    
    dict_keys= {}
    
    for x in lines:
    
        lines_1= x.split("=")
        
        dict_keys[lines_1[0]]= lines_1[1].strip() 

    return dict_keys

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
    
    # Establecer conexión 
    connection = psycopg2.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=dbname
    )
    
    cursor = connection.cursor()
    
    # Crear el cursor (en pg8000 no se necesita un cursor explícito)
    delete_query= '''DELETE FROM {} where "Datetime" >= '{}';'''.format(table_func, datetime_base)
             
    cursor.execute(delete_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()      

    return

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

def update_table_func(cripto_history_var, table_func):

    # Usar SQLAlchemy para insertar los datos
    engine= create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    
    # Usar pandas para insertar los datos en la tabla
    cripto_history_var.to_sql(table_func, engine, if_exists='append', index=False)
    
    return

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

dict_keys= keys_func()

relation= "ripple"

table= "prices_ripple"

host= dict_keys["host"]
port= dict_keys["port"]
user= dict_keys["user"]
password= dict_keys["password"]
dbname= dict_keys["dbname"]

cripto_history= generate_df_func(relation)

delete_func(cripto_history, table)

update_table_func(cripto_history, table)












#