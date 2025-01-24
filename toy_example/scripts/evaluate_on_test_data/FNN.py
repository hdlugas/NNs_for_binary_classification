
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

x_train, y_train, x_test, y_test = get_test_train_data(path='/home/hunter/KAN_toy_example/data/raw_toy_data.csv')

df_opt_params = pd.read_csv(f'/home/hunter/KAN_toy_example/data/hyperparameter_search_results/FNN_opt_params.csv')

for i in range(0,df_opt_params.shape[0]):
    num_epochs = int(df_opt_params['NUM.EPOCHS'][i])
    act_func = str(df_opt_params['ACT.FUNC'][i])
    lr = float(df_opt_params['LR'][i])
    n_neurons = int(df_opt_params['N.NEURONS'][i])

    model_ann = models.Sequential()
    if n_neurons > 0:
        model_ann.add(Dense(n_neurons, activation=act_func, input_dim=x_train.shape[1]))
        model_ann.add(Dense(1, activation='sigmoid'))
    else:
        model_ann.add(Dense(1, activation='sigmoid', input_dim=x_train.shape[1]))

    model_ann.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[keras.metrics.AUC])
    model_ann.fit(x_train, y_train, epochs=num_epochs, batch_size=None, verbose=0)

    probs = model_ann.predict(x_test).flatten()
    y_preds = [1 if prob >= 0.5 else 0 for prob in probs]

    df_out = pd.DataFrame({'PRED':y_preds, 'TRUE':y_test, 'PROB.CASE':probs})
    path_out = f'/home/hunter/KAN_toy_example/data/test_data_evaluation/FNN_num_epochs_{num_epochs}_act_func_{act_func}_lr_{lr}_n_neurons_{n_neurons}.csv'
    df_out.to_csv(path_out, index=False)


