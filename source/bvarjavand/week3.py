import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import knor
from tqdm import tqdm

def gen_d_dim_data(d, n=100):
    data = []
    mean_0 = [0]*d
    mean_1 = [np.sqrt(1/d)]*d
    cov_var = [0.5]*d
    cov = np.diag(cov_var)
    data_0 = np.random.multivariate_normal(mean_0, cov, n)
    data_1 = np.random.multivariate_normal(mean_1, cov, n)
    return data_0, data_1

def cluster(data, init0='random', clf='kmeans'):
    if clf=='kmeans' or clf=='gmm':
        if clf=='kmeans':
            classifier = KMeans(n_clusters=2, init=init0, max_iter=1)
        if clf=='kmeans-conv':
            classifier = KMeans(n_clusters=2, init=init0)
        elif clf=='gmm':
            classifier = GaussianMixture(n_components=2, init_params=init0, covariance_type='diag')
        t0 = time.time()
        classifier.fit(data)
        t1 = time.time()
        dur = t1-t0
        klabels = classifier.predict(data)
    elif clf=='knor':
        t0 = time.time()
        ret = knor.Kmeans(data, 2)
        t1 = time.time()
        dur = t1-t0
        klabels = ret.get_clusters()
    else:
        print('specify classifier')
    if np.sum(klabels[:50]) > 25:
        klabels = 1-klabels
    klabels_0 = data[klabels==0]
    klabels_1 = data[klabels==1]
    half = len(data)//2
    dlabels = np.concatenate(([0]*half,[1]*(len(data)-half)),axis=0)
    err = np.sum(((dlabels!=klabels)))/len(data)
    return klabels, klabels_0, klabels_1, err, dur

def cluster_counts(data, init0='random'):
    classifier = KMeans(n_clusters=2, init=init0)
    t0 = time.time()
    classifier.fit(data)
    t1 = time.time()
    dur = t1-t0
    klabels = classifier.predict(data)
    if np.sum(klabels[:50]) > 25:
        klabels = 1-klabels
    klabels_0 = data[klabels==0]
    klabels_1 = data[klabels==1]
    half = len(data)//2
    dlabels = np.concatenate(([0]*half,[1]*(len(data)-half)),axis=0)
    err = np.sum(((dlabels!=klabels)))/len(data)
    num_iter = classifier.n_iter_
    return klabels, klabels_0, klabels_1, err, dur, num_iter

def plot_clusters(d, data_0, data_1, klabels_0, klabels_1, clf='kmeans'):
    if d==1:
        plt.figure(figsize=(12,8))
        plt.suptitle(clf+' behavior on 1 dimension', fontsize=20)
        plt.subplot(121)
        plt.plot([0]*len(data_0) , data_0, 'r.', alpha=0.5)
        plt.plot([0]*len(data_1) , data_1, 'b.', alpha=0.5)
        plt.subplot(122)
        plt.plot([0]*len(klabels_0), klabels_0, 'r.', alpha=0.5)
        plt.plot([0]*len(klabels_1), klabels_1, 'b.', alpha=0.5)
        plt.show()
    elif d==2:
        plt.figure(figsize=(12,8))
        plt.suptitle(clf+' behavior on 2 dimensions', fontsize=20)
        plt.subplot(121)
        plt.plot(data_0[:,0], data_0[:,1], 'r.', alpha=0.5)
        plt.plot(data_1[:,0], data_1[:,1], 'b.', alpha=0.5)
        plt.subplot(122)
        plt.plot(klabels_0[:,0], klabels_0[:,1], 'r.', alpha=0.5)
        plt.plot(klabels_1[:,0], klabels_1[:,1], 'b.', alpha=0.5)
        plt.show()
    elif d==3:
        fig = plt.figure(figsize=(12,8))
        fig.suptitle(clf+' behavior on 3 dimensions', fontsize=20)
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(data_0[:,0], data_0[:,1], data_0[:,2], 'r.', alpha=0.5)
        ax.scatter(data_1[:,0], data_1[:,1], data_1[:,2], 'b.', alpha=0.5)
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(klabels_0[:,0], klabels_0[:,1], klabels_0[:,2], 'r.', alpha=0.5)
        ax.scatter(klabels_1[:,0], klabels_1[:,1], klabels_1[:,2], 'b.', alpha=0.5)
        plt.show()

def look_at_it(d, n=100, init0='random', clf='kmeans', log_err=False):
    data_0, data_1 = gen_d_dim_data(d)
    data = np.concatenate((data_0,data_1), axis=0)
    klabels, klabels_0, klabels_1, err, dur = cluster(data, init0, clf)
    print(clf+',', init0, 'initialization in',d,'dimensions')
    plot_clusters(d, data_0, data_1, klabels_0, klabels_1)
    if log_err:
        print('error:', err)

def monte(d, n=100, n_iter=100, init0='random', clf='kmeans'):
    times = []
    errors = []
    for _ in range(n_iter):
        data_0, data_1 = gen_d_dim_data(d, n)
        data = np.concatenate((data_0,data_1), axis=0)
        dlabels = np.concatenate(([0]*n,[1]*n),axis=0)
        klabels, klabels_0, klabels_1, kerr, dur = cluster(data, init0, clf)
        times.append(dur)
        errors.append(np.sum(((dlabels!=klabels)))/len(data))
    time0 = (np.mean(times), np.std(times))
    err = (np.mean(errors), np.std(errors))
    return time0, err

def monte_iters(d, n=100, n_iter=100, init0='random'):
    times = []
    errors = []
    iters = []
    for _ in range(n_iter):
        data_0, data_1 = gen_d_dim_data(d, n)
        data = np.concatenate((data_0,data_1), axis=0)
        dlabels = np.concatenate(([0]*n,[1]*n),axis=0)
        klabels, klabels_0, klabels_1, kerr, dur, iter = cluster_counts(data, init0)
        times.append(dur)
        errors.append(np.sum(((dlabels!=klabels)))/len(data))
        iters.append(iter)
    time0 = (np.mean(times), np.std(times))
    err = (np.mean(errors), np.std(errors))
    iter0 = (np.mean(iters), np.std(iters))
    return time0, err, iter0

def get_monte_data(x=[1,10,100,1000], d=1, n=100, n_iter=100, init0='random', clf='kmeans'):
    dtimes = []
    dtimes_std = []
    derrors = []
    derrors_std = []
    ntimes = []
    ntimes_std = []
    nerrors = []
    nerrors_std = []
    for val in tqdm(x):
        time0, err = monte(d, val, n_iter, init0, clf)
        ntimes.append(time0[0])
        ntimes_std.append(time0[1])
        nerrors.append(err[0])
        nerrors_std.append(err[1])
    for val in tqdm(x):
        time0, err = monte(val, n, n_iter, init0, clf)
        dtimes.append(time0[0])
        dtimes_std.append(time0[1])
        derrors.append(err[0])
        derrors_std.append(err[1])
    n_vec = (ntimes, ntimes_std, nerrors, nerrors_std)
    d_vec = (dtimes, dtimes_std, derrors, derrors_std)
    return n_vec, d_vec

def count_iters(x=[1,10,100,1000], d=1, n=100, n_iter=100, init0='random'):
    dtimes = []
    dtimes_std = []
    derrors = []
    derrors_std = []
    ntimes = []
    ntimes_std = []
    nerrors = []
    nerrors_std = []
    niters = []
    niters_std = []
    diters = []
    diters_std = []
    for val in tqdm(x):
        time0, err, iter0 = monte_iters(d, val, n_iter, init0)
        ntimes.append(time0[0])
        ntimes_std.append(time0[1])
        nerrors.append(err[0])
        nerrors_std.append(err[1])
        niters.append(iter0[0])
        niters_std.append(iter0[1])
    for val in tqdm(x):
        time0, err, iter0 = monte_iters(val, n, n_iter, init0)
        dtimes.append(time0[0])
        dtimes_std.append(time0[1])
        derrors.append(err[0])
        derrors_std.append(err[1])
        diters.append(iter0[0])
        diters_std.append(iter0[1])
    n_vec = (ntimes, ntimes_std, nerrors, nerrors_std)
    d_vec = (dtimes, dtimes_std, derrors, derrors_std)
    i_vec = (niters, niters_std, diters, diters_std)
    return n_vec, d_vec, i_vec
