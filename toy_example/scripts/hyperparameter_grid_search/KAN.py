
from sklearn.model_selection import RepeatedStratifiedKFold
from kan import *
import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from functions import *

def train_acc():
    return torch.mean((torch.argmax(model_kan(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model_kan(dataset['test_input']), dim=1) == dataset['test_label']).float())

x_train_original, y_train_original, x_test_original, y_test_original = get_test_train_data(path='/home/hunter/toy_example/data/raw_toy_data.csv')

num_epochs_range = [100]
k_range = [2]
grid_range = [2]
n_neurons_range = [5,10]

rskf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

for num_epochs in num_epochs_range:
    for k in k_range:
        for grid in grid_range:
            for n_neurons in n_neurons_range:
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
                        
                    x_train = torch.from_numpy(x_train).float()
                    x_test = torch.from_numpy(x_test).float()
                    y_train = torch.from_numpy(y_train.astype(np.int64))
                    y_test = torch.from_numpy(y_test.astype(np.int64))

                    if n_neurons == 0:
                        model_kan = KAN(width=[x_train.shape[1],2], grid=grid, k=k, seed=1)
                    else:
                        model_kan = KAN(width=[x_train.shape[1],n_neurons,2], grid=grid, k=k, seed=1)

                    dataset = {'train_input':x_train, 'train_label':y_train, 'test_input':x_test, 'test_label':y_test}
                    model_kan(dataset['train_input'])
                    results = model_kan.fit(dataset, opt='Adam', steps=num_epochs, metrics=(train_acc,test_acc), loss_fn=torch.nn.CrossEntropyLoss())
                    preds = model_kan(dataset['test_input'])
                    _, predicted_labels = torch.max(preds,1)
                    ps = softmax(preds.detach().numpy())[:,1]
                    predicted_labels = predicted_labels.numpy()
                    y_test = y_test.numpy()
                    probs = probs + ps.tolist()
                    pred_labels = pred_labels + predicted_labels.tolist()
                    true_labels = true_labels + y_test.tolist()
                    idxs = idxs + [idx] * len(ps)

                df_out = pd.DataFrame({'ITERATION':idxs, 'PRED':pred_labels, 'TRUE':true_labels, 'PROB.CASE':probs})
                path_out = f'/home/hunter/toy_example/data/hyperparameter_search_results/KAN_num_epochs_{num_epochs}_grid_{grid}_k_{k}_n_neurons_{n_neurons}.csv'
                df_out.to_csv(path_out, index=False)


