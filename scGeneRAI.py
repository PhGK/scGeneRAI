import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from itertools import permutations
import pandas as pd

from dataloading_simple import Dataset_train, Dataset_LRP
import os
from tqdm import tqdm


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return tc.mean(tc.log(tc.cosh(ey_t + 1e-12)))


class LRP_Linear(nn.Module):
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):

        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device

        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)


        zpp = self.newlayer(1).forward(Ap)  
        zmm = self.newlayer(-1, no_bias=True).forward(Am) 

        zmp = self.newlayer(1, no_bias=True).forward(Am) 
        zpm = self.newlayer(-1).forward(Ap) 

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data # new version
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sp).sum().backward()
        cmp = Am.grad

        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sm).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)


        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data


        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):

        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0) # Replace 1e-9 by zero
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0) # same here

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new


class LRP_ReLU(nn.Module):
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R



class NN(nn.Module):
    def __init__(self, inp, outp, hidden, hidden_depth):
        super(NN, self).__init__()
        self.layers = nn.Sequential(LRP_Linear(inp, hidden), LRP_ReLU())
        for i in range(hidden_depth):
            self.layers.add_module('LRP_Linear' + str(i + 1), LRP_Linear(hidden, hidden))
            self.layers.add_module('LRP_ReLU' + str(i + 1), LRP_ReLU())

        self.layers.add_module('LRP_Linear_last', LRP_Linear(hidden, outp))

    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R



class scGeneRAI:
    def __init__(self):
        pass

    def fit(self, data, nepochs, model_depth, lr=2e-2, batch_size=50, lr_decay = 0.995, descriptors = None, early_stopping = True, device_name = 'cpu'):

        self.simple_features = data.shape[1]
        if descriptors is not None:
            self.onehotter = OneHotter()
            one_hot_descriptors = self.onehotter.make_one_hot_new(descriptors)
            self.data = pd.concat([data, one_hot_descriptors], axis=1)

        else:
            self.data = data


        self.nsamples, self.nfeatures = self.data.shape
        self.hidden = 2*self.nfeatures
        self.depth = model_depth

        self.sample_names = self.data.index
        self.feature_names = self.data.columns
        self.data_tensor = tc.tensor(np.array(self.data)).float()

        self.nn = NN(2*(self.nfeatures), self.nfeatures, self.hidden, self.depth)

        tc.manual_seed(0)
        all_ids = tc.randperm(self.nsamples)
        self.train_ids, self.test_ids = all_ids[:self.nsamples//10*9], all_ids[self.nsamples//10*9:]

        testlosses, epoch_list, network_list = train(self.nn, self.data_tensor[self.train_ids], self.data_tensor[self.test_ids], nepochs, lr=lr, batch_size=batch_size,  lr_decay=lr_decay, device_name=device_name)

        if early_stopping:
            mindex = tc.argmin(testlosses)
            self.actual_testloss = testlosses[mindex]
            min_network = network_list[mindex]
            self.epochs_trained = epoch_list[mindex]
        
            self.nn = NN(2*(self.nfeatures), self.nfeatures, self.hidden, self.depth)
            self.nn.load_state_dict(min_network)
        else:
           self.epochs_trained = nepochs
           self.actual_testloss = testlosses[-1]

        print('the network trained for {} epochs (testloss: {})'.format(self.epochs_trained, self.actual_testloss))


    def predict_networks(self, data, descriptors = None, LRPau = True, remove_descriptors = True, device_name = 'cpu', PATH = '.'):
        if not os.path.exists(PATH + '/results/'):
            os.makedirs(PATH + '/results/')


        if descriptors is not None:
            one_hot_descriptors = self.onehotter.make_one_hot(descriptors)
            assert one_hot_descriptors.shape[0] == data.shape[0], 'descriptors ({}) need to have same sample size as data ({})'.format(one_hot_descriptors.shape[0],data.shape[0])
            data_extended = pd.concat([data, one_hot_descriptors], axis=1)
   
        else:
            data_extended = data

        nsamples_LRP, nfeatures_LRP = data_extended.shape
        assert nfeatures_LRP == self.nfeatures, 'neural network has been trained on {} input features, now there are  {}'.format(self.nfeatures, nfeatures_LRP)

        
        sample_names_LRP = data_extended.index
        feature_names_LRP = data_extended.columns
        data_tensor_LRP = tc.tensor(np.array(data_extended)).float()
        
        target_gene_range = self.simple_features if remove_descriptors else data_tensor_LRP.shape[1]

        for sample_id, sample_name in enumerate(sample_names_LRP):
            calc_all_paths(self.nn, data_tensor_LRP, sample_id, sample_name, feature_names_LRP, target_gene_range = target_gene_range, PATH=PATH, batch_size=100, LRPau = LRPau, device = tc.device(device_name))
        
        

class OneHotter:
    def __init__(self):
        pass

    def make_one_hot_new(self,descriptors):
        columns = []
        self.level_dict = {}
        for col in descriptors.columns:
            sel_col = descriptors[col]
            levels = sel_col.unique()
            self.level_dict[col] = levels
            one_hot = (np.array(sel_col)[:,None] == levels[None,:])*1.0
            colnames = [col + '=' + level for level in levels]
            one_hot_frame = pd.DataFrame(one_hot, columns = colnames)
            columns.append(one_hot_frame)
        return pd.concat(columns, axis=1)

    def make_one_hot(self, descriptors):
        columns = []
        for col in descriptors.columns:
            sel_col = descriptors[col]
            levels = self.level_dict[col]
            one_hot = (np.array(sel_col)[:,None] == levels[None,:])*1.0
            colnames = [col + '=' + level for level in levels]
            one_hot_frame = pd.DataFrame(one_hot, columns = colnames)
            columns.append(one_hot_frame)
        return pd.concat(columns, axis=1)
        



def train(neuralnet, train_data, test_data, epochs, lr, batch_size, lr_decay, device_name):
    device = tc.device(device_name)
    nsamples, nfeatures = train_data.shape
    optimizer = tc.optim.SGD(neuralnet.parameters(), lr=lr, momentum=0.9) 
    scheduler = ExponentialLR(optimizer, gamma = lr_decay)

    criterion = LogCoshLoss() 
    testlosses, epoch_list, network_list = [], [], []

    neuralnet.train().to(device)

    for epoch in tqdm(range(epochs)):
        if epoch<5:
            optimizer.param_groups[0]['lr']=lr/5*(epoch+1)

        trainset = Dataset_train(train_data)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

      
        for masked_data, mask, full_data in trainloader:
            masked_data = masked_data.to(device)
            mask = mask.to(device)
            full_data = full_data.to(device)


            optimizer.zero_grad()
            pred = neuralnet(masked_data)
            loss = criterion(pred[mask==0], full_data[mask==0]) 
            loss.backward()
            optimizer.step()
        scheduler.step()
            

        if epoch%10==0:
            #print(optimizer.param_groups[0]['lr'])
            neuralnet.eval()
            testset = Dataset_train(test_data)
            traintestset = Dataset_train(train_data)
            testloader = DataLoader(testset, batch_size=test_data.shape[0], shuffle=False)
            traintestloader = DataLoader(traintestset, batch_size=test_data.shape[0], shuffle=False)

            for masked_data, mask, full_data in testloader:
                masked_data = masked_data.to(device)
                mask = mask.to(device)
                full_data = full_data.to(device)
                with tc.no_grad():
                    pred = neuralnet(masked_data)
                testloss = criterion(pred[mask==0], full_data[mask==0])
                testlosses.append(testloss)
                epoch_list.append(epoch)
                network_list.append(neuralnet.state_dict())
                break

            for masked_data, mask, full_data in traintestloader:
                masked_data = masked_data.to(device)
                mask = mask.to(device)
                full_data = full_data.to(device)
                with tc.no_grad():
                    pred = neuralnet(masked_data)
                traintestloss = criterion(pred[mask==0], full_data[mask==0])
                #print(epoch, 'trainloss:', traintestloss, 'testloss:', testloss)
                break
    
    return tc.tensor(testlosses), epoch_list, network_list




def compute_LRP(neuralnet, test_set, target_id, sample_id, batch_size, device):
    criterion = nn.MSELoss()
    testset = Dataset_LRP(test_set,target_id, sample_id)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    neuralnet.to(device).eval()

    masked_data, mask, full_data = next(iter(testloader))
    masked_data, mask, full_data = masked_data.to(device), mask.to(device), full_data.to(device)
    pred = neuralnet(masked_data)

    error = criterion(pred.detach()[:,target_id], full_data.detach()[:,target_id]).cpu().numpy()
    y = full_data.detach()[:,target_id].cpu().mean().numpy()
    y_pred = pred.detach()[:,target_id].cpu().mean().numpy()

    R = tc.zeros_like(pred)
    R[:,target_id] = pred[:,target_id].clone()

    a = neuralnet.relprop(R)
    LRP_sum = (a.sum(dim=0))

    LRP_unexpanded = 0.5 * (LRP_sum[:LRP_sum.shape[0] // 2] + LRP_sum[LRP_sum.shape[0] // 2:])


    mask_sum = mask.sum(dim=0).float()

    LRP_scaled = LRP_unexpanded/mask_sum
    LRP_scaled = tc.where(tc.isnan(LRP_scaled),tc.tensor(0.0).to(device), LRP_scaled)
     
    full_data_sample = full_data[0,:].cpu().detach().numpy().squeeze()
    return LRP_scaled.cpu().numpy(), error, y , y_pred, full_data_sample


def calc_all_paths(neuralnet, test_data, sample_id, sample_name, featurenames, target_gene_range, PATH, batch_size=100, LRPau = True, device = tc.device('cpu')):
    end_frame = []

    for target in range(target_gene_range):
        LRP_value, error, y, y_pred, full_data_sample = compute_LRP(neuralnet, test_data, target, sample_id, batch_size = batch_size, device = device)

        frame = pd.DataFrame({'LRP': LRP_value[:target_gene_range], 'source_gene': featurenames[:target_gene_range], 'target_gene': featurenames[target] ,
                'sample_name': sample_name, 'error':error, 'y':y, 'y_pred':y_pred, 'inpv': full_data_sample[:target_gene_range]})
       

        end_frame.append(frame)
        end_result_path = PATH + '/results/'  + 'LRP_' + str(sample_id) + '_'+ str(sample_name) + '.csv'
        #if not os.path.exists(result_path + data_type):
        #    os.makedirs(result_path + data_type)

    end_frame = pd.concat(end_frame, axis=0)

    if LRPau:
        end_frame_re = end_frame.copy()
        end_frame_re['LRP_abs_re'] = np.abs(end_frame_re['LRP']) 
        end_frame_re = end_frame_re[['LRP_abs_re', 'source_gene', 'target_gene']]
        end_frame_kontra = end_frame_re.rename(columns = {'LRP_abs_re': 'LRP_abs_kontra','source_gene': 'target_gene', 'target_gene': 'source_gene'})

        end_frame_au = end_frame_re.merge(end_frame_kontra)

        end_frame_au['LRP'] = 0.5 * (end_frame_au['LRP_abs_re'] + end_frame_au['LRP_abs_kontra'])    

        end_frame = end_frame_au.copy()[['LRP', 'source_gene', 'target_gene']]
        end_frame = end_frame[end_frame['source_gene']>end_frame['target_gene']]



    end_frame.to_csv(end_result_path)





