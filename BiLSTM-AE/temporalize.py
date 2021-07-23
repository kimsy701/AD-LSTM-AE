#load data and temporalize to windows
import numpy as np
import pandas as pd

# x, y
train = pd.read_csv('split/inmun_train.csv')
x_train = train.loc[:,['UsedPower']] #2차원 dataframe으로
y_train = train.loc[:,['label']]

val = pd.read_csv('split/inmun_val.csv')
x_val = val.loc[:,['UsedPower']] #2차원 dataframe으로
y_val = val.loc[:,['label']]

test=pd.read_csv('split/inmun_test.csv')
x_test = test.loc[:,['UsedPower']]
y_test = test.loc[:,['label']]

#n_features = input_x.shape[1]

#transform to time series data #lostm is 3d data (samples, time series, feature)
def sum_list(input_list,i,j): #2,3.4.5 #2,5
    #total sum = list[i] +list[i+1]+...list[j]
    total_sum=0
    for k in range(j-i+1): #48 #0,1,2...48
        total_sum += input_list.iloc[i+k] #2,3,4,5
        total_sum=total_sum.values
    return total_sum

timesteps = 48

# Temporalize train data
output_X = []
output_y = []
for i in range(48, x_train.shape[0]):
    output_X.append(x_train.iloc[i-48:i, 0])


for i in range(x_train.shape[0]-timesteps):
    if sum_list(y_train, i, i+timesteps-1)==0: #y,84105,84152
        y_window= 0
    else:
        y_window = 1
    output_y.append(y_window)

output_y = np.array(output_y)
output_X,output_y = np.array(output_X), np.array(output_y)

output_X = np.reshape(output_X, (len(x_train)-timesteps, timesteps,1))
final_output_X = np.squeeze(output_X, axis=2)
final_output_X.shape
output_X=pd.DataFrame(final_output_X)
output_y=pd.DataFrame(output_y)

output_X.to_csv('inmun_train_x.csv')
output_y.to_csv('inmun_train_y.csv')

# Temporalize val data
output_X = []
output_y = []
for i in range(48, x_val.shape[0]):
    output_X.append(x_val.iloc[i-48:i, 0])


for i in range(x_val.shape[0]-timesteps):
    if sum_list(y_val, i, i+timesteps-1)==0: #y,84105,84152
        y_window= 0
    else:
        y_window = 1
    output_y.append(y_window)

output_y = np.array(output_y)
output_X,output_y = np.array(output_X), np.array(output_y)

output_X = np.reshape(output_X, (len(x_val)-timesteps, timesteps,1))
final_output_X = np.squeeze(output_X, axis=2)
final_output_X.shape
output_X=pd.DataFrame(final_output_X)
output_y=pd.DataFrame(output_y)

output_X.to_csv('inmun_val_x.csv')
output_y.to_csv('inmun_val_y.csv')


# Temporalize test data
output_X = []
output_y = []
for i in range(48, x_test.shape[0]):
    output_X.append(x_test.iloc[i-48:i, 0])


for i in range(x_test.shape[0]-timesteps):
    if sum_list(y_test, i, i+timesteps-1)==0: #y,84105,84152
        y_window= 0
    else:
        y_window = 1
    output_y.append(y_window)



output_y = np.array(output_y)
output_X,output_y = np.array(output_X), np.array(output_y)

output_X = np.reshape(output_X, (len(x_test)-timesteps, timesteps,1))
final_output_X = np.squeeze(output_X, axis=2)
final_output_X.shape
output_X=pd.DataFrame(final_output_X)
output_y=pd.DataFrame(output_y)

output_X.to_csv('inmun_test_x.csv')
output_y.to_csv('inmun_test_y.csv')