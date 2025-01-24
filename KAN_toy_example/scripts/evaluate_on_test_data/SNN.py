
from sklearn.model_selection import RepeatedStratifiedKFold
import snntorch as snn
from snntorch import functional as SF
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from functions import *

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

x_train, y_train, x_test, y_test = get_test_train_data(path='/home/hunter/KAN_toy_example/data/raw_toy_data.csv')

x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

df_opt_params = pd.read_csv(f'/home/hunter/KAN_toy_example/data/hyperparameter_search_results/SNN_opt_params.csv')

for i in range(0,df_opt_params.shape[0]):
    num_epochs = int(df_opt_params['NUM.EPOCHS'][i])
    beta = float(df_opt_params['BETA'][i])
    num_steps = int(df_opt_params['NUM.STEPS'][i])
    n_neurons = int(df_opt_params['N.NEURONS'][i])
    correct_rate = float(df_opt_params['CORRECT.RATE'][i])
    lr = float(df_opt_params['LR'][i])

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
        probs = tmp[:,1]
        y_preds = [1 if prob >= 0.5 else 0 for prob in probs]

    df_out = pd.DataFrame({'PRED':y_preds, 'TRUE':y_test, 'PROB.CASE':probs})
    path_out = f'/home/hunter/KAN_toy_example/data/test_data_evaluation/SNN_num_epochs_{num_epochs}_beta_{beta}_num_steps_{num_steps}_n_neurons_{n_neurons}_correct_rate_{correct_rate}_lr_{lr}.csv'
    df_out.to_csv(path_out, index=False)


