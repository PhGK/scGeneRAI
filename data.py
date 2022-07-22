import pandas as pd
import torch as tc
import numpy as np
import numpy.ma as ma

def load_data(name):
    normalize_data = False

    if name == 'random':
        randomized_data = tc.randn(1000,10)
        sample_names = ['sample' + str(i) for i in range(1000)]
        feature_names = ['feature' + str(i) for i in range(10)]

    if normalize_data:
        meanv, sdv, minv, maxv = randomized_data.mean(axis=0, keepdim=True), randomized_data.std(axis=0, keepdim=True), \
                                 randomized_data.min(axis=0)[0], tc.abs(randomized_data).max(axis=0)[0]
        


        randomized_data = (randomized_data) / sdv

    dataframe = pd.DataFrame(randomized_data.numpy())
    dataframe.columns = np.array(feature_names)
    dataframe.index = np.array(sample_names)
    return dataframe




