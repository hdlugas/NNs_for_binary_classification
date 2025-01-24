
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

def scale_data(x, mu="default", sigma="default", return_mu_sigma=False):
    # function adapted from Mendez KM, Reinke SN, Broadhurst DI. A comparative evaluation of the generalised predictive ability of eight machine learning algorithms across ten clinical metabolomics data sets for binary classification. Metabolomics. 2019 Nov 15;15(12):150. doi: 10.1007/s11306-019-1612-4. PMID: 31728648; PMCID: PMC6856029.
    # https://github.com/CIMCB/MetabComparisonBinaryML/tree/master
    x = np.array(x)

    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)

    if mu is "default":
        mu = np.nanmean(x, axis=0)
    if sigma is "default":
        sigma = np.nanstd(x, axis=0, ddof=1)
        sigma = np.where(sigma==0, 1, sigma)  # if a value in sigma equals 0 it is converted to 1

    z = (x - mu) / sigma

    if return_mu_sigma is True:
        return z, mu, sigma
    else:
        return z



def knnimpute(x, k=3):
    # function adapted from Mendez KM, Reinke SN, Broadhurst DI. A comparative evaluation of the generalised predictive ability of eight machine learning algorithms across ten clinical metabolomics data sets for binary classification. Metabolomics. 2019 Nov 15;15(12):150. doi: 10.1007/s11306-019-1612-4. PMID: 31728648; PMCID: PMC6856029.
    # https://github.com/CIMCB/MetabComparisonBinaryML/tree/master

    # Tranpose x so we treat columns as features, and rows as samples
    x = x.T

    # Error check for k value
    k_max = x.shape[1] - 1
    if k_max < k:
        raise ValueError("k value is too high. Max k value is {}".format(k_max))

    # z is the returned array with NaNs imputed
    z = x.copy()

    # Use columns without NaNs for knnimpute
    nan_check = np.isnan(x)
    no_nan = np.where(sum(nan_check.T) == 0, 1, 0)

    # Error check that not all columns have NaNs
    x_no_nan = x[no_nan == 1]
    if x_no_nan.size == 0:
        raise ValueError("All colummns of the input data contain missing values. Unable to impute missing values.")

    # Calculate pairwise distances between columns, and covert to square-form distance matrix
    pair_dist = pdist(x_no_nan.T, metric="euclidean")
    sq_dist = squareform(pair_dist)

    # Make diagonals negative and sort
    dist = np.sort(sq_dist - np.eye(sq_dist.shape[0], sq_dist.shape[1])).T
    dist_idx = np.argsort(sq_dist - np.eye(sq_dist.shape[0], sq_dist.shape[1])).T

    # Find where neighbours are equal distance
    equal_dist_a = np.diff(dist[1:].T, 1, 1).T == 0
    equal_dist_a = equal_dist_a.astype(int)  # Convert to integer
    equal_dist_b = np.zeros(len(dist))
    equal_dist = np.concatenate((equal_dist_a, [equal_dist_b]))  # Concatenate

    # Get rows and cols for missing values
    nan_idx = np.argwhere(nan_check)
    nan_rows = nan_idx[:, 0]
    nan_cols = nan_idx[:, 1]

    # Make sure rows/cols are in a list (note: this happens when there is 1 missing value)
    if type(nan_rows) is not np.ndarray:
        nan_rows = [nan_rows]
        nan_cols = [nan_cols]

    # Impute each NaN value
    for i in range(len(nan_rows)):

        # Error check for rows with all NaNs
        if np.isnan(x[nan_rows[i], :]).all() == True:
            warnings.warn("Row {} contains all NaNs, so Row {} is imputed with zeros.".format(nan_rows[i], nan_rows[i]), Warning)

        # Create a loop from 1 to len(dist_idx) - k
        lastk = len(dist_idx) - k
        loopk = [1]
        while lastk > loopk[-1]:
            loopk.append(loopk[-1] + 1)

        # Impute
        for j in loopk:
            L_a = equal_dist[j + k - 2 :, nan_cols[i]]
            L = np.where(L_a == 0)[0][0]  # equal_dist neighbours

            x_vals_r = nan_rows[i]
            x_vals_c = dist_idx[j : j + k + L, nan_cols[i]]
            x_vals = x[x_vals_r, x_vals_c]
            weights = 1 / dist[1 : k + L + 1, nan_cols[i]]
            imp_val = wmean(x_vals, weights)  # imputed value
            if imp_val is not np.nan:
                z[nan_rows[i], nan_cols[i]] = imp_val
                break

    # Transpose z
    z = z.T
    z[np.isnan(z)] = 0
    return z



def wmean(x, weights):
    # function adapted from Mendez KM, Reinke SN, Broadhurst DI. A comparative evaluation of the generalised predictive ability of eight machine learning algorithms across ten clinical metabolomics data sets for binary classification. Metabolomics. 2019 Nov 15;15(12):150. doi: 10.1007/s11306-019-1612-4. PMID: 31728648; PMCID: PMC6856029.
    # https://github.com/CIMCB/MetabComparisonBinaryML/tree/master

    # Flatten x and weights
    x = x.flatten()
    weights = weights.flatten()

    # Find NaNs
    nans = np.isnan(x)
    infs = np.isinf(weights)

    # If all x are nans, return np.nan
    if nans.all() == True:
        m = np.nan
        return m

    # If there are infinite weights, use the corresponding x
    if infs.any() == True:
        m = np.nanmean(x[infs])
        return m

    # Set NaNs to zero
    x[nans] = 0
    weights[nans] = 0
    
    # Normalize the weights + calculate Weighted Mean
    weights = weights / np.sum(weights)
    m = np.matmul(weights, x)
    return m



def get_test_train_data(path):
    df = pd.read_csv(path)
    outcomes = df['Class']
    y = outcomes.values
    df = df.drop(columns=['Class'])

    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=1/3, stratify=y, random_state=3)
    x_train = np.log(x_train+1)
    x_train, mu, sigma = scale_data(x_train, return_mu_sigma=True)
    x_train = knnimpute(x_train, k=3)

    x_test = np.log(x_test+1)
    x_test = scale_data(x_test, mu=mu, sigma=sigma)
    x_test = knnimpute(x_test, k=3)
    return x_train, y_train, x_test, y_test



def softmax(x):
    ex = np.exp(x - np.max(x,axis=1,keepdims=True))
    return ex / ex.sum(axis=1,keepdims=True)

