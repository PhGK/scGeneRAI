from scGeneRAI import scGeneRAI
#from dataloading_simple import Dataset_train, Dataset_LRP
from dataloading_simple import Dataset_train, Dataset_LRP
from data import load_data
import torch as tc
import sys
from joblib import Parallel, delayed
import os



PATH = '.'
model_path = PATH + '/results/models/'
result_path= PATH + '/results/LRP_values_raw/'


rand_data = load_data('random')

predictor = scGeneRAI()

predictor.fit(rand_data, nepochs = 200, lr = 1e-4, model_depth = 2)


