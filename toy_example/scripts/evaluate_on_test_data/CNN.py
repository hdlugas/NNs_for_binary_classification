
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras import layers, models
import keras
import torch
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from functions import *

x_train, y_train, x_test, y_test = get_test_train_data(path='/home/hunter/toy_example/data/raw_toy_data.csv')

dropout_rate = 0.5

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

df_opt_params = pd.read_csv(f'/home/hunter/toy_example/data/hyperparameter_search_results/CNN_opt_params.csv')

for i in range(0,df_opt_params.shape[0]):
    num_epochs = int(df_opt_params['NUM.EPOCHS'][i])
    act_func = str(df_opt_params['ACT.FUNC'][i])
    lr = float(df_opt_params['LR'][i])
    n_neurons1 = int(df_opt_params['N.NEURONS1'][i])
    n_neurons2 = int(df_opt_params['N.NEURONS2'][i])
    pool_size = int(df_opt_params['POOL.SIZE'][i])
    kernel_size = int(df_opt_params['KERNEL.SIZE'][i])

    model = models.Sequential()
    model.add(layers.Input(shape=(x_train.shape[1],1)))
    model.add(layers.Conv1D(filters=n_neurons1, kernel_size=kernel_size, activation=act_func))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.Conv1D(filters=n_neurons2, kernel_size=kernel_size, activation=act_func))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_neurons2, activation=act_func))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_neurons1, activation=act_func))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[keras.metrics.AUC])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=None, verbose=0)
    probs = model.predict(x_test).flatten()
    y_preds = [1 if prob >= 0.5 else 0 for prob in probs]

    df_out = pd.DataFrame({'PRED':y_preds, 'TRUE':y_test, 'PROB.CASE':probs})
    path_out = f'/home/hunter/toy_example/data/test_data_evaluation/CNN_num_epochs_{num_epochs}_act_func_{act_func}_lr_{lr}_n_neurons1_{n_neurons1}_n_neurons2_{n_neurons2}_pool_size_{pool_size}_kernel_size_{kernel_size}.csv'
    df_out.to_csv(path_out, index=False)


