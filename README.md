# GeneRAI<sup>sc
  
```python
  from scGeneRAI import scGeneRAI
  
```
  initialize model and fit data
```
  model = scGeneRAI()
  model.fit(data, model_depth, nepochs, lr=2e-2, batch_size=50, lr_decay = 0.995, descriptors = None, early_stopping = True, device_name = 'cpu')
  
```
- data: A pandas dataframe consisting of shape m x n, containing RNA samples of m cells and n genes.
- nepochs: Number of training epochs
- model_depth: (default=2)
- lr: learning rate of stochastic gradient descent optimizer (default=2e-2)
- batch_size (default=50)
- lr_decay: Learning rate decay using pytorch exponential learning rate scheduler (lr_decay orresponds to pythorch's gamma, default: 0.995)
- descriptors: Pandas frame of additional **categorical** cell descriptors, e.g. batch, cell type. Need to have the same sample size as *data*  (default=None)
- early_stopping: If True, scGeneRAI chooses the model with the smallest test loss during training (default=True).
- device_name: can be used to run computation on GPU (e.g. with 'cuda:0', default='cpu'). device_name is give to torch.device().
  
  
  predict networks 
```
  model.predict_networks(...)
```
