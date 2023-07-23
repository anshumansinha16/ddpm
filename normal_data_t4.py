import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data_dssi as ds
import pandas as pd

def read_normal_data():

    data = ds.read_data()

    print(data.head())


    first_entry = data.iloc[0]
    print(first_entry)
    print(first_entry[0].shape)
    print(first_entry[1].shape)
    exit()

    # Column 1 (leads)

    #print(data.iloc[:,0][0].shape) # (500, 12)
    li_1 = data['leads'].tolist()
    num_sub = len(li_1)
    li_1 = np.vstack(li_1)

    scalar_list = []

    final_dat = np.empty(li_1.shape)

    for i in range(li_1.shape[1]):
        scaler = MinMaxScaler()
        b = li_1[:,i].reshape(-1,1)
        normalized_data = scaler.fit_transform(b)
        scalar_list.append(scaler)
        final_dat[:,i] = normalized_data[:,0]

    #
    #print(final_dat.shape)
    a_f = np.vsplit(final_dat, num_sub)
    
    #print(a_f[0].shape)

    # Column 2 (act)

    li_2 = data['act'].tolist() # 3x500x75 -> 1x1000
    print('length li_2',len(li_2))
    print(li_2[0].shape)
    li_2 = [i.reshape(1,-1) for i in li_2]
    #print(np.array(li_2).reshape(num_sub,1,-1).shape)
    print('li_2',li_2)
    li_2 = np.vstack(li_2)
    print('li_2',li_2.shape)

    scaler = MinMaxScaler()
    normalized_data_label = scaler.fit_transform(li_2)

    print('normalized_data_label',normalized_data_label.shape)
    a_f2 = np.vsplit(normalized_data_label, num_sub)
    a_f2 = [i.reshape(500,-1) for i in a_f2]
    #print(a_f2[0].shape)

    df = pd.DataFrame({'leads': a_f, 'act': a_f2})

    return df



    #print(df.head())

    #print('------------------------------')

    #first_entry = df.iloc[0]
    #print(first_entry)
    #print(first_entry[1].dtype)



    



