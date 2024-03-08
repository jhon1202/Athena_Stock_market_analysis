#%%
import numpy as np
import pandas as pd
import math
import operator
import heapq
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statistics
from sklearn.model_selection import train_test_split

df_close = pd.read_csv('c:/ATENA/ASSETS/SUPERIOR/TheWorld.csv', usecols=[5], header= None)
df_date = pd.read_csv('c:/ATENA/ASSETS/SUPERIOR/TheWorld.csv', usecols=[0], header= None).values
df_time = pd.read_csv('c:/ATENA/ASSETS/SUPERIOR/TheWorld.csv', usecols=[1], header= None).values
df_asset = pd.read_csv('c:/ATENA/ASSETS/SUPERIOR/TheWorld.csv', usecols=[7], header= None).values

Data_close = df_close.values
Rate1_step = int(len(df_close)/4)
Rate1_arr = []
for i in range(Rate1_step):
    Rate1 = Data_close[i*4+3][0]*100/Data_close[i*4][0]-100
    #Rate1 = math.floor(Rate1*1000)/1000
    Rate1_arr.append(Rate1)
#print(Rate1_arr)
Rate2_arr = []

for j in range(len(Rate1_arr)-1):
    Rate2 = Rate1_arr[j+1]*100/Rate1_arr[j]-100
    #if Rate2 >=120:
        #var_arr = [Data_close[j*4][0], Data_close[j*4+1][0], Data_close[j*4+2][0], Data_close[j*4+3][0]]
        #ex = [1,2,3,4]
        #max_point = max(var_arr)
    #else:
        #max_point = 0
    #Rate2 = math.floor(Rate2*1000)/1000
    Rate2_arr.append(Rate2)
    #max_arr.append(max_point)
#print(Rate2_arr)
No_arr = []
index_arr = []
for k in range(len(Rate2_arr)):
    if Rate2_arr[k] >= 120:
        var_max_arr = [Data_close[(k+1)*4][0], Data_close[(k+1)*4+1][0], Data_close[(k+1)*4+2][0], Data_close[(k+1)*4+3][0]]
        max_index, max_value = max(enumerate(var_max_arr), key=operator.itemgetter(1))        
        #max_point = 0
        #for m in range(0, 4):
            #if Data_close[(k+1)*4+m][0] > max_point:
                #max_point = Data_close[(k+1)*4+m][0]
        No_arr.append(max_value)
        index_arr.append((k+1)*4 + max_index)
    elif Rate2_arr[k] <= 56:
        var_min_arr = [Data_close[(k+1)*4][0], Data_close[(k+1)*4+1][0], Data_close[(k+1)*4+2][0], Data_close[(k+1)*4+3][0]]
        min_index, min_value = min(enumerate(var_min_arr), key=operator.itemgetter(1))
        No_arr.append(min_value)
        index_arr.append((k+1)*4 + min_index)
        
#print(index_arr)
#print(No_arr)
Axis_X_arr = []
Axis_Y_arr = []

for m in range(len(index_arr)-1):
    Axis_X = index_arr[m+1]-index_arr[m]
    Axis_Y = No_arr[m+1]-No_arr[m]+No_arr[m]
    Axis_X_arr.append(Axis_X)
    Axis_Y_arr.append(math.floor(Axis_Y*100)/100)
#print(Axis_X_arr)
#print(Axis_Y_arr)
AX_AY_corresponding_date_times = []
for n in range(len(Axis_X_arr)):
    AX_corresponding_date = df_date[index_arr[n]:index_arr[n+1]]
    AX_corresponding_time = df_time[index_arr[n]:index_arr[n+1]]
    AX_corresponding_date_time_arr = []
    for p in range(len(AX_corresponding_date)):
        AX_corresponding_date_time = AX_corresponding_date[p] + '-' + AX_corresponding_time[p]
        AX_corresponding_date_time_arr.append(AX_corresponding_date_time)
    AX_AY_corresponding_date_times.append([Axis_X_arr[n], Axis_Y_arr[n], AX_corresponding_date_time_arr]) 
    
pd.DataFrame(AX_AY_corresponding_date_times).to_csv("AX_AY_Date.csv")
    
    


Baseline_AX_arr = [2, 4, 7, 4, 3, 2, 7, 3, 5, 1, 6, 5]
#print(Baseline_AX_arr)

correlation_number_X_arr=[]
for n in range(len(Axis_X_arr)-11):
    Gen_AX_arr= [Axis_X_arr[n], Axis_X_arr[n+1], Axis_X_arr[n+2], Axis_X_arr[n+3],  Axis_X_arr[n+4],  Axis_X_arr[n+5],  Axis_X_arr[n+6],  Axis_X_arr[n+7],  Axis_X_arr[n+8],  Axis_X_arr[n+9],  Axis_X_arr[n+10],  Axis_X_arr[n+11]]
    x_train, x_test, y_train, y_test = train_test_split(Baseline_AX_arr, Gen_AX_arr, test_size=0.2, random_state=0)
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    n= len(y_test)
    p = x_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    correlation_number = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    correlation_number_X_arr.append(correlation_number)
#print('-----------correlation_number_X_arr --------------')
#print(correlation_number_X_arr),  Axis_X_arr[n+11]

X = heapq.nlargest(10, correlation_number_X_arr)
X_indexes = heapq.nlargest(10, range(len(correlation_number_X_arr)), key=correlation_number_X_arr.__getitem__)
#print(X)
#print(X_indexes)

Baseline_AY_arr = [1671.1, 1659.77, 1694.75, 1666.01, 1651.85, 1627.97, 1633.46, 1629.47, 1755.37, 1770.71, 1737.6, 1741.39]
#print(Baseline_AY_arr)

correlation_number_Y_arr=[]
for n in range(len(Axis_Y_arr)-11):
    Gen_AY_arr= [Axis_Y_arr[n], Axis_Y_arr[n+1], Axis_Y_arr[n+2], Axis_Y_arr[n+3], Axis_Y_arr[n+4], Axis_Y_arr[n+5], Axis_Y_arr[n+6], Axis_Y_arr[n+7], Axis_Y_arr[n+8], Axis_Y_arr[n+9], Axis_Y_arr[n+10], Axis_Y_arr[n+11] ]
    x_train, x_test, y_train, y_test = train_test_split(Baseline_AY_arr, Gen_AY_arr, test_size=0.2, random_state=0)
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    n= len(y_test)
    p = x_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    correlation_number = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    correlation_number_Y_arr.append(correlation_number)
#print('-----------correlation_number_Y_arr --------------')
#print(correlation_number_Y_arr)

Y = heapq.nlargest(10, correlation_number_Y_arr)
Y_indexes = heapq.nlargest(10, range(len(correlation_number_Y_arr)), key=correlation_number_Y_arr.__getitem__)
#print(Y)
#print(Y_indexes)

correlation_XY_arr = []
for l in range(len(correlation_number_Y_arr)):
    XY_val = statistics.mean([correlation_number_Y_arr[l], correlation_number_X_arr[l]])
    correlation_XY_arr.append(XY_val)

XY = heapq.nlargest(10, correlation_XY_arr)
XY_indexes = heapq.nlargest(10, range(len(correlation_XY_arr)), key=correlation_XY_arr.__getitem__)
#print(XY)
#print(XY_indexes)
X_corresponding_date_times = []

for h in range(0, 5):
    index_inf = math.floor((index_arr[X_indexes[h]]+1)/4)*4
    index_sup = (math.floor((index_arr[X_indexes[h]]+1)/4)+13)*4
    X_corresponding_date = df_date[index_inf:index_sup]
    X_corresponding_time = df_time[index_inf:index_sup]
    X_corresponding_date_time_arr = []
    for p in range(len(X_corresponding_date)):
        X_corresponding_date_time = X_corresponding_date[p] + '-' + X_corresponding_time[p]
        X_corresponding_date_time_arr.append(X_corresponding_date_time)
    X_corresponding_date_times.append([X[h], X_corresponding_date_time_arr])
print('-----------X values and Times --------------')
print(X_corresponding_date_times)

Y_corresponding_date_times = []

for h in range(0, 5):
    index_inf = math.floor((index_arr[Y_indexes[h]]+1)/4)*4
    index_sup = (math.floor((index_arr[Y_indexes[h]]+1)/4)+13)*4
    Y_corresponding_date = df_date[index_inf:index_sup]
    Y_corresponding_time = df_time[index_inf:index_sup]
    Y_corresponding_date_time_arr = []
    for p in range(len(Y_corresponding_date)):
        Y_corresponding_date_time = Y_corresponding_date[p] + '-' + Y_corresponding_time[p]
        Y_corresponding_date_time_arr.append(Y_corresponding_date_time)
    Y_corresponding_date_times.append([Y[h], Y_corresponding_date_time_arr])
    
print('-----------Y values and Times --------------')
print(Y_corresponding_date_times)

XY_corresponding_date_times = []

for h in range(0, 7):
    index_inf = math.floor((index_arr[XY_indexes[h]]+1)/4)*4
    index_sup = (math.floor((index_arr[XY_indexes[h]]+1)/4)+13)*4
    XY_corresponding_asset = df_asset[index_inf:index_sup]
    XY_corresponding_date = df_date[index_inf:index_sup]
    XY_corresponding_time = df_time[index_inf:index_sup]
    XY_corresponding_date_time_arr = []
    for p in range(len(XY_corresponding_date)):
        XY_corresponding_date_time = XY_corresponding_date[p] + '-' + XY_corresponding_time[p] + '-' + XY_corresponding_asset[p]
        XY_corresponding_date_time_arr.append(XY_corresponding_date_time)
    XY_corresponding_date_times.append([XY[h], XY_corresponding_date_time_arr])
    
print('-----------XY values and Times --------------')
print(XY_corresponding_date_times)

# %%
