
from sklearn.model_selection import RepeatedStratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras import models
import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from functions import *

x_train_original, y_train_original, x_test_original, y_test_original = get_test_train_data(path='/home/hunter/toy_example/data/raw_toy_data.csv')

num_epochs_range = [100]
lr_range = [0.01]
n_neurons_range = [5,10]
act_funcs = ['sigmoid']

rskf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

for num_epochs in num_epochs_range:
    for lr in lr_range:
        for n_neurons in n_neurons_range:
            for act_func in act_funcs:
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
                        
                    model_ann = models.Sequential()
                    if n_neurons > 0:
                        model_ann.add(Dense(n_neurons, activation=act_func, input_dim=x_train.shape[1]))
                        model_ann.add(Dense(1, activation='sigmoid'))
                    else:
                        model_ann.add(Dense(1, activation='sigmoid', input_dim=x_train.shape[1]))

                    model_ann.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[keras.metrics.AUC])
                    model_ann.fit(x_train, y_train, epochs=num_epochs, batch_size=None, verbose=0)
                    probs_tmp = model_ann.predict(x_test).flatten()
                    y_preds = [1 if prob >= 0.5 else 0 for prob in probs_tmp]

                    probs = probs + probs_tmp.tolist()
                    pred_labels = pred_labels + y_preds
                    true_labels = true_labels + y_test.tolist()
                    idxs = idxs + [idx] * len(probs_tmp)

                df_out = pd.DataFrame({'ITERATION':idxs, 'PRED':pred_labels, 'TRUE':true_labels, 'PROB.CASE':probs})
                path_out = f'/home/hunter/toy_example/data/hyperparameter_search_results/FNN_num_epochs_{num_epochs}_act_func_{act_func}_lr_{lr}_n_neurons_{n_neurons}.csv'
                df_out.to_csv(path_out, index=False)


