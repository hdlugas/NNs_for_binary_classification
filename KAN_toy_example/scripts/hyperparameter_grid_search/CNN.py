
from functions import *
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras import layers, models
import keras
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

x_train_original, y_train_original, x_test_original, y_test_original = get_test_train_data(path='/home/hunter/KAN_toy_example/data/raw_toy_data.csv')

num_epochs_range = [20]
lr_range = [0.01]
n_neurons1_range = [16]
n_neurons2_range = [32]
pool_size_range = [2]
dropout_rate_range = [0.5]
kernel_size_range = [2]
act_funcs = ['sigmoid','relu']

rskf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

for num_epochs in num_epochs_range:
    for lr in lr_range:
        for n_neurons1 in n_neurons1_range:
            for n_neurons2 in n_neurons2_range:
                for pool_size in pool_size_range:
                    for dropout_rate in dropout_rate_range:
                        for kernel_size in kernel_size_range:
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

                                    x_train = x_train[..., np.newaxis]
                                    x_test = x_test[..., np.newaxis]

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
                                    probs_tmp = model.predict(x_test).flatten()
                                    y_preds = [1 if prob >= 0.5 else 0 for prob in probs_tmp]

                                    probs = probs + probs_tmp.tolist()
                                    pred_labels = pred_labels + y_preds
                                    true_labels = true_labels + y_test.tolist()
                                    idxs = idxs + [idx] * len(probs_tmp)

                                df_out = pd.DataFrame({'ITERATION':idxs, 'PRED':pred_labels, 'TRUE':true_labels, 'PROB.CASE':probs})
                                path_out = f'/home/hunter/KAN_toy_example/data/hyperparameter_search_results/CNN_num_epochs_{num_epochs}_act_func_{act_func}_lr_{lr}_n_neurons1_{n_neurons1}_n_neurons2_{n_neurons2}_pool_size_{pool_size}_dropout_rate_{dropout_rate}_kernel_size_{kernel_size}.csv'
                                df_out.to_csv(path_out, index=False)




