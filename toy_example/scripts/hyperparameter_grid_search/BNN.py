
from functions import *
import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold

x_train_original, y_train_original, x_test_original, y_test_original = get_test_train_data(path='/home/hunter/KAN_toy_example/data/raw_toy_data.csv')

num_epochs_range = [20]
lr_range = [0.01]
n_neurons_range = [5]
kl_weight_range = [0.01]
act_funcs = ['relu','sigmoid']

rskf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

for num_epochs in num_epochs_range:
    for lr in lr_range:
        for n_neurons in n_neurons_range:
            for kl_weight in kl_weight_range:
                for act_func in act_funcs:
                    idx = -1
                    idxs = []
                    true_labels = []
                    pred_labels = []
                    probs = []
                    for train_idx, test_idx in rskf_cv.split(x_train_original, y_train_original):
                        x_train, x_test = x_train_original[train_idx], x_train_original[test_idx]
                        y_train, y_test = y_train_original[train_idx], y_train_original[test_idx]
                        idx = idx + 1
                        print(f'\nIteration #{idx}')

                        x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
                        x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()

                        if act_func == 'relu':
                            model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x_train.shape[1], out_features=n_neurons), nn.ReLU(), bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_neurons, out_features=2),)
                        elif act_func == 'sigmoid':
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
                            probs_tmp = tmp[:,1]
                            y_preds = [1 if prob >= 0.5 else 0 for prob in probs_tmp]

                            probs = probs + probs_tmp.tolist()
                            pred_labels = pred_labels + y_preds
                            true_labels = true_labels + y_test.tolist()
                            idxs = idxs + [idx] * len(probs_tmp)

                    df_out = pd.DataFrame({'ITERATION':idxs, 'PRED':pred_labels, 'TRUE':true_labels, 'PROB.CASE':probs})
                    path_out = f'/home/hunter/KAN_toy_example/data/hyperparameter_search_results/BNN_num_epochs_{num_epochs}_lr_{lr}_n_neurons_{n_neurons}_act_func_{act_func}_kl_weight_{kl_weight}.csv'
                    df_out.to_csv(path_out, index=False)


