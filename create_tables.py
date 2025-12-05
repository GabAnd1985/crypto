# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:13:57 2024

@author: Gabriel
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine

Ruta= "D:/Cripto"

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

def keys_func():
    
    fileref= open("{}/keys_Master.txt".format(Ruta), "r")
    
    lines= fileref.readlines()
    
    dict_keys= {}
    
    for x in lines:
    
        lines_1= x.split("=")
        
        dict_keys[lines_1[0]]= lines_1[1].strip() 

    return dict_keys

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Función para crear la tabla precios

def create_prices_func(name):

    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    
    cursor = connection.cursor()
    
    create_table1_query = '''
    CREATE TABLE IF NOT EXISTS {} (
        "Datetime" TIMESTAMP,
        "Close" NUMERIC(20,10),
        "Volume" BIGINT,
        "Day" TEXT
    );
    '''.format(name)
    
    cursor.execute(create_table1_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()
    
    return 

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Función para crear la tabla models

def create_models_func(name):

    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    
    cursor = connection.cursor()
    
    create_table2_query = '''
    CREATE TABLE IF NOT EXISTS {} (
        "Datetime" TIMESTAMP,
        "Model_version" TEXT,
        "Date_train_test" DATE,     
        "Date_start_predict" DATE,
        "mape_1d" NUMERIC(20,16),
        "mape_1d_str" TEXT,
        "mape_2d" NUMERIC(20,16),
        "mape_2d_str" TEXT
    );
    '''.format(name)
    
    cursor.execute(create_table2_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()
    
    return

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Función para crear la tabla prediction

def create_prediction_func(name):

    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    
    cursor = connection.cursor()
    
    create_table3_query = '''
    CREATE TABLE IF NOT EXISTS {} (
        "Date_train_test" DATE,  
        "Model_version" TEXT,
        "Datetime" TIMESTAMP,   
        "predicted_price" NUMERIC(20,10),
        "real_price" NUMERIC(20,10)
    );
    '''.format(name)
    
    cursor.execute(create_table3_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()
    
    return    

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Función para apendar precios históricos (CUIDADO BORRA TODA LA TABLA)

def append_historical_prices(archivo, tabla):

    df_1= pd.read_csv("{}/Archivos Soporte/{}.csv".format(Ruta, archivo))
    
    df= df_1.rename(columns={'Day_of_Week': 'Day'}) 
    
    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    
    cursor = connection.cursor()
    
    delete_query = '''DELETE FROM {};'''.format(tabla)
    
    cursor.execute(delete_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()
    
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))
    
    df.to_sql('{}'.format(tabla), engine, if_exists='append', index=False)

    return 

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

# Generar un usuario con SELECT e INSERT permisos

def New_user_func(new_user, new_password):
    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )

    cursor = connection.cursor()

    # Create the new user
    try:
        cursor.execute(f"CREATE USER {new_user} WITH PASSWORD '{new_password}';")
        print(f"User '{new_user}' successfully created.")
    except psycopg2.errors.DuplicateObject:
        print(f"The user '{new_user}' already exists.")

    # Grant permissions
    try:
        # Grant basic database and schema access
        cursor.execute(f"GRANT CONNECT ON DATABASE {dbname} TO {new_user};")
        cursor.execute(f"GRANT USAGE ON SCHEMA public TO {new_user};")

        # Grant SELECT and INSERT permissions on existing tables
        cursor.execute(f"GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO {new_user};")

        # Ensure future tables also have these permissions
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT ON TABLES TO {new_user};")

        print(f"Permissions (SELECT, INSERT) granted to '{new_user}'.")
    except Exception as e:
        print("Error while granting permissions:", e)

    # Confirm changes and close connection
    connection.commit()
    cursor.close()
    connection.close()

    return

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

# Generar un usuario con SELECT / INSERT / DELETE permisos

def New_user_delete_func(new_user, new_password):
    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )

    cursor = connection.cursor()

    # Create the new user
    try:
        cursor.execute(f"CREATE USER {new_user} WITH PASSWORD '{new_password}';")
        print(f"User '{new_user}' successfully created.")
    except psycopg2.errors.DuplicateObject:
        print(f"The user '{new_user}' already exists.")

    # Grant permissions
    try:
        # Grant basic database and schema access
        cursor.execute(f"GRANT CONNECT ON DATABASE {dbname} TO {new_user};")
        cursor.execute(f"GRANT USAGE ON SCHEMA public TO {new_user};")

        # Grant SELECT, INSERT, and DELETE permissions on existing tables
        cursor.execute(f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA public TO {new_user};")

        # Ensure future tables also have these permissions (SELECT, INSERT, DELETE)
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, DELETE ON TABLES TO {new_user};")

        print(f"Permissions (SELECT, INSERT, DELETE) granted to '{new_user}'.")
    except Exception as e:
        print("Error while granting permissions:", e)

    # Confirm changes and close connection
    connection.commit()
    cursor.close()
    connection.close()

    return

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

#Ejecutar la creación de la tabla precios

dict_keys= keys_func()

host= dict_keys["host"]
port= dict_keys["port"]
user= dict_keys["user"]
password= dict_keys["password"]
dbname= dict_keys["dbname"]

#create_prices_func("prices_ripple")

#Ejecutar la creación de la tabla models

#create_models_func("models_ripple")

#Ejecutar la creación de la tabla prediction

#create_prediction_func("prediction_ripple")

#Ejecutar el acople de precios históricos (CUIDADO BORRA TODA LA TABLA)

#append_historical_prices("ethereum_history", "prices_ethereum")

#append_historical_prices("cardano_history", "prices_cardano")

#append_historical_prices("solana_history", "prices_solana")

#append_historical_prices("ripple_history", "prices_ripple")

#Ejecutar el código que me permite crear un usuario con SELECT e INSERT permisos

#New_user_func("", "")

#Ejecutar el código que me permite crear un usuario con SELECT / INSERT / DELETE permisos

#New_user_delete_func("", "")






#