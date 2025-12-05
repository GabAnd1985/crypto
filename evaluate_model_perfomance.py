# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:34:58 2024

@author: Gabriel
"""

from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import inspect
import os
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.layers import Bidirectional
from tensorflow.keras.losses import Huber

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def keys_func():

    Ruta= "D:/Cripto"
    
    fileref= open("{}/keys.txt".format(Ruta), "r")
    
    lines= fileref.readlines()
    
    dict_keys= {}
    
    for x in lines:
    
        lines_1= x.split("=")
        
        dict_keys[lines_1[0]]= lines_1[1].strip() 

    return dict_keys

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def definicion_rutas_func(cripto_to_analize):

    Ruta= "D:/Cripto/Comparativa_Modelos/{}".format(cripto_to_analize)
    
    #Ruta_OneDrive= "C:/Users/anima/OneDrive/Gabriel One Drive/Cripto/Comparativa_Modelos/{}".format(cripto_to_analize)

    return Ruta #, Ruta_OneDrive

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def prepare_input_function(date_train_test, timestep, cripto_to_analize):

    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))
    
    query_1 = 'SELECT * FROM "prices_{}";'.format(cripto_to_analize)
    
    cripto_history_3= pd.read_sql_query(query_1, engine)
    
    #Dividir entre train and test
    
    cripto_history_4= cripto_history_3[["Datetime", "Close"]]
    
    cripto_history_5= cripto_history_4.copy()
    
    cripto_history_5["param"]= date_train_test
    
    cripto_history_5['param']= pd.to_datetime(cripto_history_5['param'])
    
    cripto_train_1= cripto_history_5[cripto_history_5["Datetime"] < cripto_history_5["param"]]
    
    cripto_train_2= cripto_train_1.drop(["param"], axis= "columns")
    
    cripto_train= cripto_train_2.sort_values(by= "Datetime")
    
    cripto_test_1= cripto_history_5[cripto_history_5["Datetime"] >= cripto_history_5["param"]]
    
    cripto_test_2= cripto_test_1.drop(["param"], axis= "columns")
    
    cripto_test= cripto_test_2.sort_values(by= "Datetime")
    
    scaler= MinMaxScaler()
    
    df1= cripto_train['Close']
    
    train_array= scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    df2= cripto_test['Close']
    
    test_array= scaler.transform(np.array(df2).reshape(-1,1))
    
    time_step = timestep
    
    dataX,dataY = [],[]
    
    for i in range(len(train_array)-time_step-1):
                   a = train_array[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(train_array[i + time_step,0])
                   X_train_cripto= np.array(dataX)
                   Y_train_cripto= np.array(dataY)
    
    dataX,dataY = [],[]
    
    for i in range(len(test_array)-time_step-1):
                   a = test_array[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(test_array[i + time_step,0])
                   X_test_cripto= np.array(dataX)
                   Y_test_cripto= np.array(dataY)

    return cripto_train, cripto_test, scaler, X_train_cripto, Y_train_cripto, X_test_cripto, Y_test_cripto

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v1():
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True,input_shape = (X_train_cripto.shape[1],1)))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error',optimizer = 'adam')
    
    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v2():

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train_cripto.shape[1], 1)))  # Más unidades
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(25))  # Menos unidades en la última capa
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v3():

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_cripto.shape[1], 1)))
    model.add(Dropout(0.2))  # Regularización
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v4():

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_cripto.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    # Cambiar el optimizador a RMSprop
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v5():

    model = Sequential()
    model.add(LSTM(50,return_sequences = True,input_shape = (X_train_cripto.shape[1],1)))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss=Huber(delta=1.0), optimizer='adam')
    
    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v6():

    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train_cripto.shape[1], 1)))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 100,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def train_model_func_v7():
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True,input_shape = (X_train_cripto.shape[1],1)))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error',optimizer = 'adam')
    
    model.fit(X_train_cripto,Y_train_cripto,validation_data = (X_test_cripto,Y_test_cripto),epochs = 50,batch_size = 64,verbose = 1)

    return model

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def predict_Func(model):

    #train_predict_cripto= model.predict(X_train_cripto)
    
    #first_100_rows= X_test_cripto[:101, :]  #Extraigo las últimas 100 filas del array
    
    #X_test_cripto_adjusted= np.vstack((first_100_rows, X_test_cripto))
    
    test_predict_cripto= model.predict(X_test_cripto)
    
    # transform to original form
    
    #train_predict_cripto= scaler.inverse_transform(train_predict_cripto)
    
    test_predict_cripto= scaler.inverse_transform(test_predict_cripto)
    
    #Prepare data for plotting
    
    cripto_train_full= cripto_train.set_index('Datetime')
    
    cripto_test_full= cripto_test.set_index('Datetime')
    
    test_predict_cripto_series= pd.Series(test_predict_cripto.flatten(), index= cripto_test_full[101:].index)

    return cripto_train_full, cripto_test_full, test_predict_cripto_series

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def save_graphs_func(model_version, cripto_train_full, cripto_test_full, test_predict_cripto_series):

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(cripto_train_full["Close"], label='Training Data')
    plt.plot(cripto_test_full["Close"], label='Test Data', color='orange')
    plt.plot(test_predict_cripto_series, label='Predicted Data', color='green')
    
    # Añadir título y etiquetas
    plt.title('LSTM - Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Añadir leyenda
    plt.legend()
    
    plt.savefig('{}/{}/{}/Price_Pred_lg_{}.png'.format(Ruta, date_train_test, model_version, model_version), dpi=300, bbox_inches='tight')

    #plt.savefig('{}/{}/{}/Price_Pred_lg_{}.png'.format(Ruta_OneDrive, date_train_test, model_version, model_version), dpi=300, bbox_inches='tight')
    
    #-----------------------------------------------------------------------
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(cripto_train_full["Close"].tail(100), label='Training Data')
    plt.plot(cripto_test_full["Close"].head(200), label='Test Data', color='orange')
    plt.plot(test_predict_cripto_series.head(100), label='Predicted Data', color='green')
    
    # Añadir título y etiquetas
    plt.title('LSTM - Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Añadir leyenda
    plt.legend()
    
    plt.savefig('{}/{}/{}/Price_Pred_sg_{}.png'.format(Ruta, date_train_test, model_version, model_version), dpi=300, bbox_inches='tight')

    #plt.savefig('{}/{}/{}/Price_Pred_sg_{}.png'.format(Ruta_OneDrive, date_train_test, model_version, model_version), dpi=300, bbox_inches='tight')

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def save_model_code_func(train_model_func, model_version):

    codigo_func= inspect.getsource(train_model_func)
    
    with open("{}/{}/{}/Model_Code_{}.txt".format(Ruta, date_train_test, model_version, model_version), "w") as archivo:
        archivo.write(codigo_func)

    #with open("{}/{}/{}/Model_Code_{}.txt".format(Ruta_OneDrive, date_train_test, model_version, model_version), "w") as archivo:
    #    archivo.write(codigo_func)

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def generar_ruta_func(model_version):

    ruta_directorio = "{}/{}/{}".format(Ruta, date_train_test, model_version)
    
    os.makedirs(ruta_directorio, exist_ok=True)  

    #ruta_directorio_1 = "{}/{}/{}".format(Ruta_OneDrive, date_train_test, model_version)
    
    #os.makedirs(ruta_directorio_1, exist_ok=True)

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def evaluate_model(test_predict_cripto_series, cripto_test_full):

    predict_1= test_predict_cripto_series.reset_index()
    
    predict_2= predict_1.rename(columns={0: 'Predict'})
    
    predict_2['Date'] = predict_2['Datetime'].dt.date
    
    Date_1= predict_2[["Date"]].drop_duplicates()
    
    start_predict_1= Date_1.head(1)

    start_predict_2= start_predict_1.reset_index()

    date_start_predict= start_predict_2["Date"][0]
    
    #----------------------------------------------------------------
    
    #1 day
    
    Date_2= Date_1.head(1)
    
    Date_3= Date_2.copy()
    
    Date_3["Marca"]= True
    
    predict_3= pd.merge(predict_2, Date_3, on= "Date", how= "left")
    
    predict_4= predict_3[predict_3["Marca"]== True]
    
    predict_5= predict_4[["Datetime", "Predict"]]
    
    Compare_1= pd.merge(predict_5, cripto_test_full, on= "Datetime", how= "left")
    
    mape_1d= mean_absolute_percentage_error(Compare_1["Close"], Compare_1["Predict"]) * 100
    
    mape_1d_str= f"{mape_1d:.4f}%"
    
    #----------------------------------------------------------------
    
    #2 days
    
    Date_2= Date_1.head(2)
    
    Date_3= Date_2.copy()
    
    Date_3["Marca"]= True
    
    predict_3= pd.merge(predict_2, Date_3, on= "Date", how= "left")
    
    predict_4= predict_3[predict_3["Marca"]== True]
    
    predict_5= predict_4[["Datetime", "Predict"]]
    
    Compare_real_vs_pred= pd.merge(predict_5, cripto_test_full, on= "Datetime", how= "left")
    
    mape_2d= mean_absolute_percentage_error(Compare_real_vs_pred["Close"], Compare_real_vs_pred["Predict"]) * 100
    
    mape_2d_str= f"{mape_2d:.4f}%"

    return date_start_predict, Compare_real_vs_pred, mape_1d, mape_1d_str, mape_2d, mape_2d_str

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def insert_model_perf_func(model_version, date_start_predict, mape_1d, mape_1d_str, mape_2d, mape_2d_str, cripto_to_analize):

    # Conexión a la base de datos
    conn = psycopg2.connect(
        dbname= dbname,
        user= user,
        password= password,
        host= host,  
        port= port
    )
    
    # Crear un cursor
    cur = conn.cursor()
    
    delete_query = '''DELETE FROM "models_{}" where ("Date_train_test"= '{}' and "Model_version"= '{}');'''.format(cripto_to_analize, date_train_test, model_version)

    cur.execute(delete_query)

    conn.commit()
    
    # Definir la consulta de inserción
    insert_query = '''
    INSERT INTO "models_{}" (
        "Datetime", 
        "Model_version", 
        "Date_train_test", 
        "Date_start_predict", 
        "mape_1d", 
        "mape_1d_str", 
        "mape_2d", 
        "mape_2d_str"
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    '''.format(cripto_to_analize)
    
    # Datos para insertar
    registro = (
        datetime.now(),       #Fecha en que se corrió el modelo
        model_version,              #Versión de modelo          
        date_train_test,      #Fecha en que se dividen los datos de test vs train       
        date_start_predict,   #Fecha en que se comienza a predecir         
        mape_1d,              #Formato float - exactitud a 1 día
        mape_1d_str,          #Formato str % - exactitud a 1 día
        mape_2d,              #Formato float - exactitud a 2 días
        mape_2d_str           #Formato str % - exactitud a 2 días            
    )
    
    # Ejecutar la consulta de inserción
    cur.execute(insert_query, registro)
    
    # Confirmar los cambios
    conn.commit()
    
    # Cerrar el cursor y la conexión
    cur.close()
    conn.close()

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def save_model_perf_func(model_version, date_start_predict, mape_1d, mape_1d_str, mape_2d, mape_2d_str):

    with open("{}/{}/{}/Model_Performance_{}.txt".format(Ruta, date_train_test, model_version, model_version), 'w') as file:
        file.write("Resultados del Modelo\n")
        file.write("=" * 30 + "\n\n")
        file.write(f"Modelo {model_version}\n")
        file.write(f"Fecha de inicio de predicción: {date_start_predict}\n")
        file.write(f"MAPE 1D: {mape_1d} ({mape_1d_str})\n")
        file.write(f"MAPE 2D: {mape_2d} ({mape_2d_str})\n")
        file.write("-" * 30 + "\n")
    
    #with open("{}/{}/{}/Model_Performance_{}.txt".format(Ruta_OneDrive, date_train_test, model_version, model_version), 'w') as file:
    #    file.write("Resultados del Modelo\n")
    #    file.write("=" * 30 + "\n\n")
    #    file.write(f"Modelo {model_version}\n")
    #    file.write(f"Fecha de inicio de predicción: {date_start_predict}\n")
    #    file.write(f"MAPE 1D: {mape_1d} ({mape_1d_str})\n")
    #    file.write(f"MAPE 2D: {mape_2d} ({mape_2d_str})\n")
    #    file.write("-" * 30 + "\n")

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def save_trained_model_func(model, model_version):

    model.save("{}/{}/{}/trained_model_{}.keras".format(Ruta, date_train_test, model_version, model_version))
    
    #model.save("{}/{}/{}/trained_model_{}.keras".format(Ruta_OneDrive, date_train_test, model_version, model_version)) 

    return

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def insert_predicted_prices_func(compare_real_vs_pred, model_version, cripto_to_analize):

    prices_predict_1= compare_real_vs_pred.copy()
    
    prices_predict_1["Model_version"]= model_version
    
    prices_predict_1["Date_train_test"]= date_train_test
    
    prices_predict_2= prices_predict_1.rename(columns= {"Predict": "predicted_price", "Close": "real_price"})
    
    prices_predict= prices_predict_2[["Date_train_test", "Model_version", "Datetime", "predicted_price", "real_price"]]
    
    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    
    cursor = connection.cursor()
    
    delete_query = '''DELETE FROM "prediction_{}" where ("Date_train_test"= '{}' and "Model_version"= '{}');'''.format(cripto_to_analize, date_train_test, model_version)
    
    cursor.execute(delete_query)
    
    connection.commit()
    
    cursor.close()
    
    connection.close()
    
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))
    
    prices_predict.to_sql('prediction_{}'.format(cripto_to_analize), engine, if_exists='append', index=False)

    return 

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def workflow_func(train_model_func, model_version):
    
    model= train_model_func()
    
    cripto_train_full, cripto_test_full, test_predict_cripto_series = predict_Func(model)
    
    generar_ruta_func(model_version)
    
    save_graphs_func(model_version, cripto_train_full, cripto_test_full, test_predict_cripto_series)
    
    save_model_code_func(train_model_func, model_version)
    
    date_start_predict, compare_real_vs_pred, mape_1d, mape_1d_str, mape_2d, mape_2d_str= evaluate_model(test_predict_cripto_series, cripto_test_full)
    
    insert_model_perf_func(model_version, date_start_predict, mape_1d, mape_1d_str, mape_2d, mape_2d_str, cripto_to_analize)
    
    save_model_perf_func(model_version, date_start_predict, mape_1d, mape_1d_str, mape_2d, mape_2d_str)
    
    save_trained_model_func(model, model_version)
    
    insert_predicted_prices_func(compare_real_vs_pred, model_version, cripto_to_analize)
    
    return 

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

# Parámetros de la corrida

#Seteo el date_train_test que es la fecha a partir de la cual separo lo que
#es entrenamiento de test. Hasta esta fecha es entrenamiento, luego sirve para 
#proyectar.

date_list= ["2025-10-17", "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21"]

#Criptos a trabajar

list_to_analize= ["bitcoin", "cardano", "ethereum", "ripple", "solana"]

#Modelos a entrenar

dict_versions= {"v5": train_model_func_v5}

timestep= 100 #Observaciones para atrás que vamos a usar para predecir

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

# Ejecución

dict_keys= keys_func()

host= dict_keys["host"]
port= dict_keys["port"]
user= dict_keys["user"]
password= dict_keys["password"]
dbname= dict_keys["dbname"]

for date_to in date_list:
    
    print("")
    
    print(date_to)
    
    date_train_test= date_to
    
    for cripto_to in list_to_analize:
    
        print("")
    
        print(cripto_to)
        
        cripto_to_analize= cripto_to
        
        for version_to in dict_versions:
               
            print("")
            
            print("Model {}".format(version_to))
            
            #Ruta, Ruta_OneDrive= definicion_rutas_func(cripto_to_analize)
            
            Ruta= definicion_rutas_func(cripto_to_analize)
            
            cripto_train, cripto_test, scaler, X_train_cripto, Y_train_cripto, X_test_cripto, Y_test_cripto= prepare_input_function(date_train_test, timestep, cripto_to_analize)
            
            print("")  
            
            workflow_func(dict_versions[version_to], model_version= "{}".format(version_to))





    
#