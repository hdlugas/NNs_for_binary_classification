
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

x_train, y_train, x_test, y_test = get_test_train_data(path='/home/hunter/toy_example/data/raw_toy_data.csv')

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train.astype(np.int64))
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test.astype(np.int64))

df_opt_params = pd.read_csv(f'/home/hunter/toy_example/data/hyperparameter_search_results/KAN_opt_params.csv')

for i in range(0,df_opt_params.shape[0]):
    num_epochs = int(df_opt_params['NUM.EPOCHS'][i])
    grid = int(df_opt_params['GRID'][i])
    k = int(df_opt_params['K'][i])
    n_neurons = int(df_opt_params['N.NEURONS'][i])

    if n_neurons == 0:
        model_kan = KAN(width=[x_train.shape[1],2], grid=grid, k=k, seed=1)
    else:
        model_kan = KAN(width=[x_train.shape[1],n_neurons,2], grid=grid, k=k, seed=1)

    dataset = {'train_input':x_train, 'train_label':y_train, 'test_input':x_test, 'test_label':y_test}
    model_kan(dataset['train_input'])
    results = model_kan.fit(dataset, opt='Adam', steps=num_epochs, metrics=(train_acc,test_acc), loss_fn=torch.nn.CrossEntropyLoss())
    preds = model_kan(dataset['test_input'])
    _, predicted_labels = torch.max(preds,1)
    probs = softmax(preds.detach().numpy())[:,1]
    pred_labels = predicted_labels.numpy().tolist()

    df_out = pd.DataFrame({'PRED':pred_labels, 'TRUE':y_test.numpy(), 'PROB.CASE':probs})
    path_out = f'/home/hunter/toy_example/data/test_data_evaluation/KAN_num_epochs_{num_epochs}_grid_{grid}_k_{k}_n_neurons_{n_neurons}.csv'
    df_out.to_csv(path_out, index=False)


