# scGeneRAI

Please cite [Keyl et al](https://academic.oup.com/nar/article/51/4/e20/6984592?login=false) 

```python
  from scGeneRAI import scGeneRAI
  
```
  initialize model and fit data
```
  model = scGeneRAI()
  model.fit(data, model_depth, nepochs, lr=2e-2, batch_size=5, lr_decay = 0.99, descriptors = None, early_stopping = True, device_name = 'cpu')
  
```
- data: A pandas dataframe with shape m x n, containing RNA samples of m cells and n genes.
- nepochs: Number of training epochs
- model_depth: (default=2)
- lr: learning rate of stochastic gradient descent optimizer (default=2e-2)
- batch_size (default=5)
- lr_decay: Learning rate decay using pytorch exponential learning rate scheduler (lr_decay corresponds to pytorch's gamma, default: 0.99)
- descriptors: Pandas frame of additional **categorical** cell descriptors, e.g. batch, cell type. Need to have the same sample size as *data*  (default=None)
- early_stopping: If True, scGeneRAI chooses the model with the smallest test loss during training (default=True).
- device_name: can be used to run computation on GPU (e.g. with 'cuda:0', default='cpu'). device_name is give to torch.device().
  
  
  predict networks 
```
  model.predict_networks(data, descriptors = None, LRPau = True, remove_descriptors = True, device_name = 'cpu', PATH = '.')
```
  
- data: A pandas dataframe with shape k x n, containing RNA samples of k cells and n genes. While this dataframe may contain an arbitrary number of samples, gene should exactly match the genes on which the model was fitted.
  
- descriptors: Pandas frame of additional **categorical** cell descriptors, e.g. batch, cell type. Need to have the same sample size as *data*  (default=None). Also, describing features must have been seen during **model.fit(...)**.
  
- LRPau: If *True*, predict_networks() returns the absolute undirected LRP scores between every pair of genes (default=True)
- remove_descriptors: If *True*, interactions between data features and descriptor features are omitted (default=True)
- device_name: Can be used to run computation on GPU (e.g. with 'cuda:0', default='cpu'). device_name is handed to torch.device().
- PATH: PATH defines the location where locations will be saved. (default = '.')
  
