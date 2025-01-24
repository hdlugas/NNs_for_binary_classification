
import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold
import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from functions import *

x_train, y_train, x_test, y_test = get_test_train_data(path='/home/hunter/toy_example/data/raw_toy_data.csv')

x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()

df_opt_params = pd.read_csv(f'/home/hunter/toy_example/data/hyperparameter_search_results/BNN_opt_params.csv')

for i in range(0,df_opt_params.shape[0]):
    num_epochs = int(df_opt_params['NUM.EPOCHS'][i])
    lr = float(df_opt_params['LR'][i])
    n_neurons = int(df_opt_params['N.NEURONS'][i])
    kl_weight = float(df_opt_params['KL.WEIGHT'][i])
    act_func = str(df_opt_params['ACT.FUNC'][i])

    if act_func == 'relu':
        model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x_train.shape[1], out_features=n_neurons), nn.ReLU(), bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_neurons, out_features=2),)
    else:
        model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x_train.shape[1], out_features=n_neurons), nn.Sigmoid(), bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_neurons, out_features=2),)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(0,num_epochs):
        pre = model(x_train)
        ce = ce_loss(pre, y_train)
        kl = kl_loss(model)
        cost = ce + kl_weight*kl
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        _, predicted = torch.max(pre.data, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(x_test).numpy()
        tmp = softmax(outputs)
        probs = tmp[:,1]
        y_preds = [1 if prob >= 0.5 else 0 for prob in probs]

    df_out = pd.DataFrame({'PRED':y_preds, 'TRUE':y_test, 'PROB.CASE':probs})
    path_out = f'/home/hunter/toy_example/data/test_data_evaluation/BNN_num_epochs_{num_epochs}_lr_{lr}_n_neurons_{n_neurons}_act_func_{act_func}_kl_weight_{kl_weight}.csv'
    df_out.to_csv(path_out, index=False)


