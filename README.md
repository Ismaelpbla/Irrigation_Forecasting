
# IRRIGATION FORECASTING WITH TIME SERIES

## OBJECTIVE

The aim of the project is to automate the irrigation of an orange grove by predicting the temperature, salinity and humidity of the soil.



## Authors

- [@Ismael P. Blanquer](https://www.github.com/Ismaelpbla)


## DATA

The data were obtained from a sensor located in the centre of the orange grove.
## DATA VISUALIZATION

The characteristics that can be observed in the data visualisation are :

- The temperature and salinity values show a very large variability (noise) in the first centimetres of depth. 
- From high depth values (between 50 and 80 cm) this variability is reduced and a clear trend can be observed.
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/T10.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/T30.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/T50.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/T80.png?raw=true)
- In the case of humidity, the noise is more evident in the warmer months (July - August) and also decreases with depth. This is due to the fact that during the summer months there is constant irrigation until September, when irrigation stops and harvesting begins.
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/H50.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/H60.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/H70.png?raw=true)
- The tendency of both salinity and temperature is to increase during the summer months and decrease during the winter months.
- The shallower layers are more affected by the outside temperature, which is why the first 20 cm of soil depth are generally dispensed with.
- On the other hand, the soil is practically isothermal at depths of more than 70 cm, which is why this layer is usually omitted as well.
- In the humidity graph, a very evident peak can be observed at depths between 60 and 70 cm, corresponding to a torrential rainfall.

### Seasonality
Two types of seasonality are observed:
- Diurnal and nocturnal variation is observed in temperature and salinity values.
- As there are only 6 months of data, the downward trend of temperature and salinity values can be seen. However, if we had a whole year, we could see the seasonality between the cold and warm months.

In the case of humidity, the seasonality exists during the summer months when there is irrigation, and disappears from September onwards when harvesting begins. This can be seen both in the graph and in the autocorrelation plot.
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/AUCP_inv2.png?raw=true)
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/AUCP_ver.png?raw=true)

### Is the serie stationary?

To check whether the temperature, salinity and humidity series were stationary, we tested them with the Füller index. We found that:
- Both the values of temperature, salinity and humidity in winter, had values above 0.05 and therefore were STATIONARY series.
- The humidity in summer has a value lower than 0.05, so it is a NON-STATIONARY series.


## MACHINE LEARNING MODELING

### Random Forest Regressor.
We are going to try a Regression model such as Random Forest, to see if we can improve our model. First we are going to transform the temperature, salinity and humidity series to be able to work with this model. To do this we are going to apply a shift 12 times

```
for i in range(12,0,-1):
    temperatura_pro['t-'+str(i)] = temperatura_pro['T'].shift(i)

temp_clean = temperatura_pro.dropna()

temp_clean
```

Now we define X and Y and divide them into train and test. Remember that we wanted to know the forecast for the next 3 days, i.e. 72 hours.

```
X_temp = temp_clean.loc[:, 't-12':]
y_temp = temp_clean.loc[:, 'T']

X_sal = sal_clean.loc[:, 's-12':]
y_sal = sal_clean.loc[:, 'S']

X_hum = hum_clean.loc[:, 't-12':]
y_hum = hum_clean.loc[:, 'H']

```

Train the model:
```
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=8)

rf_model.fit(X_train, y_train)

```

### ARIMA
We will take the values corresponding to the last 72 hours for the test values, and the rest of the values for the train values.

```
X_train_temperatura = temperatura_pro[:-72]
X_test_temperatura = temperatura_pro[-72:]

X_train_salinidad = salinidad_pro[:-72]
X_test_salinidad = salinidad_pro[-72:]

print('X_train_temperatura shape :', X_train_temperatura.shape)
print('X_test_temperatura shape :', X_test_temperatura.shape)
print('X_train_salinidad shape :', X_train_salinidad.shape)
print('X_test_salinidad shape :', X_test_salinidad.shape)

X_train_temperatura

```

We will run the Autoarima model with p and q values between 1 and 5 and a maximum value of d = 3, since the series is stationary. 

```
modelT = auto_arima(X_train_temperatura,
                    start_p=1,
                    start_q=1,
                    max_d=3,
                    max_p=5,
                    max_q=5,
                    trace=True,
                    stepwise=True)
```

We do the same with humidity and salinity values.

### LSTM

We choose the steps to perform LSTM, in this case we will use one step = 12 hours.

```
temperatura_pro_2 = temperatura_pro.copy()
emb_size = 12

'''
Montamos nuevas features con los lags
'''
for i in range(1, emb_size+1):
    temperatura_pro_2['lag' + str(i)] = temperatura_pro_2['T'].shift(i)
    
temperatura_pro_2.dropna(inplace=True)
temperatura_pro_2.reset_index(drop=True, inplace=True)

values = temperatura_pro_2.values
```

```
'''
Volvemos a montar xtrain, xtest...
'''
trainX,trainY = values[0:4220-emb_size ,1:],values[0:4220-emb_size ,0],
testX,testY = values[4220-emb_size:4292-emb_size,1:], values[4220-emb_size:4292-emb_size,0]

print("Train data length:", trainX.shape)
print("Train target length:", trainY.shape)
print("Test data length:", testX.shape)
print("Test target length:", testY.shape)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape

```

Model assamble

```
'''
embedding es la cantidad de lags utilizada
'''
def build_simple_rnn(num_units=128, embedding=12,num_dense=32,lr=0.001):
    """
    Builds and compiles a simple RNN model
    Arguments:
              num_units: Number of units of a the simple RNN layer
              embedding: Embedding length
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              lr: Learning rate (uses RMSprop optimizer)
    Returns:
              A compiled Keras model.
    """
    model = Sequential()
    # Long short term memory
    # Esto es capa de entrada + capa con 128 neuronas con su función de activacion
    model.add(LSTM(units=num_units, input_shape=(1,13), activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                  #optimizer=RMSprop(lr=lr),
                  optimizer='adam',
                  metrics=['mse'])
    
    return model
```

Finally we train the model:

```
batch_size=16
num_epochs = 50

model.fit(train,train, 
          epochs=num_epochs, 
          batch_size=batch_size, 
          verbose=0)

```
## MODEL PREDICTIONS

### Random Forest Regression

#### Temperature
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/RF_temp.png?raw=true)

#### Salinity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/RF_sal.png?raw=true)

#### Humidity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/RF_hum.png?raw=true)

### ARIMA

#### Temperature
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/pred_arima_temp.png?raw=true)

#### Salinity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/pred_ARIMA_sal.png?raw=true)

#### Humidity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/pred_hum.png?raw=true)

### LSTM

#### Temperature
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/temp_LSTM.png?raw=true)

#### Salinity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/sal_LSTM.png?raw=true)

#### Humidity
![img](https://github.com/Ismaelpbla/Irrigation_Forecasting/blob/main/imagenes/hum_LSTM.png?raw=true)


## DISCUSSION AND CONCLUSIONS

This project was a practical part of the Bridge Bootcamp of Data Science in Valencia. All the conclusions and discussions of the data were in the presentation of this project, also included in this repository.
Check the PDF presentation for more information.