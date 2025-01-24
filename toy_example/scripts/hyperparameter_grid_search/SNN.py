
from functions import *
from sklearn.model_selection import RepeatedStratifiedKFold
import snntorch as snn
from snntorch import functional as SF
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

class net_snn(torch.nn.Module):
    def __init__(self, timesteps, hidden, beta):
        super().__init__() 
        self.timesteps = timesteps
        self.hidden = hidden
        self.beta = beta

        self.fc1 = torch.nn.Linear(in_features=x_train.shape[1], out_features=self.hidden)
        self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden)

        self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=2)
        self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=2)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
    
        spk_recording = []
        for step in range(self.timesteps):
            spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
            spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
            spk_recording.append(spk2)
        return torch.stack(spk_recording)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

x_train_original, y_train_original, x_test_original, y_test_original = get_test_train_data(path='/home/hunter/KAN_toy_example/data/raw_toy_data.csv')

num_epochs_range = [10]
beta_range = [0.75]
num_steps_range = [10]
n_neurons_range = [3,5]
correct_rate_range = [0.8]
lr_range = [0.01]
batch_size = 1

rskf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

for num_epochs in num_epochs_range:
    for beta in beta_range:
        for num_steps in num_steps_range:
            for n_neurons in n_neurons_range:
                for correct_rate in correct_rate_range:
                    for lr in lr_range:
                        idx = -1
                        idxs = []
                        true_labels = []
                        pred_labels = []
                        probs = []
                        for train_idx, test_idx in rskf_cv.split(x_train_original, y_train_original):
                            idx = idx + 1
                            print(f'\nIteration #{idx}')

                            x_train, x_test = x_train_original[train_idx], x_train_original[test_idx]
                            y_train, y_test = y_train_original[train_idx], y_train_original[test_idx]
                                        
                            x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
                            x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()

                            train_dataset = TensorDataset(x_train, y_train)
                            test_dataset = TensorDataset(x_test, y_test)
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

                            model = net_snn(timesteps=num_steps, hidden=n_neurons, beta=beta).to(device)

                            loss_function = SF.mse_count_loss(correct_rate=correct_rate, incorrect_rate=1-correct_rate)
                            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
                            loss_hist = []
                            with trange(num_epochs) as pbar:
                                for _ in pbar:
                                    train_batch = iter(train_loader)
                                    minibatch_counter = 0
                                    loss_epoch = []
                                    for feature, label in train_batch:
                                        feature = feature.to(device)
                                        label = label.to(device)
                                        spk = model(feature.flatten(1))
                                        loss_val = loss_function(spk, label)
                                        optimizer.zero_grad()
                                        loss_val.backward()
                                        optimizer.step()
                                        loss_hist.append(loss_val.item())
                                        minibatch_counter += 1
                                        avg_batch_loss = sum(loss_hist) / minibatch_counter
                                        pbar.set_postfix(loss="%.3e" % avg_batch_loss)


                            test_batch = iter(test_loader)
                            minibatch_counter = 0
                            loss_epoch = []

                            model.eval()
                            with torch.no_grad():
                                outputs = model(x_test).numpy()
                                tmp = np.sum(outputs,axis=0) / outputs.shape[0]
                                tmp = softmax(tmp)
                                probs_tmp = tmp[:,1]
                                y_preds = [1 if prob >= 0.5 else 0 for prob in probs_tmp]

                                probs = probs + probs_tmp.tolist()
                                pred_labels = pred_labels + y_preds
                                true_labels = true_labels + y_test.tolist()
                                idxs = idxs + [idx] * len(probs_tmp)

                        df_out = pd.DataFrame({'ITERATION':idxs, 'PRED':pred_labels, 'TRUE':true_labels, 'PROB.CASE':probs})
                        path_out = f'/home/hunter/KAN_toy_example/data/hyperparameter_search_results/SNN_num_epochs_{num_epochs}_beta_{beta}_num_steps_{num_steps}_n_neurons_{n_neurons}_correct_rate_{correct_rate}_lr_{lr}.csv'
                        df_out.to_csv(path_out, index=False)



