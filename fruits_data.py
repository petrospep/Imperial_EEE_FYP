import numpy as np

def prepare_data(n,order):

    x,y=np.load('data/train_data.npy'), np.load('data/test_data.npy')
    data = np.append(y, x, axis=0)
    test_data = data[0:n, :]/255
    traind = data[n:, :]/255
    test_data = test_data.reshape((test_data.shape[0], 100, 100, 3), order=order)
    traind = traind.reshape((traind.shape[0], 100, 100, 3), order=order)
    return traind,test_data


