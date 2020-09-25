import numpy as np
import pandas as pd
import pickle
import os

root_dir = '/home/renyi/Documents/multi-level-vae/'

# fixed variance, mean generated differently
def generate(T, var, theta, n):
    llimit = int(T * 0.1)
    rlimit = int(T * 0.9)
    etas = np.array(range(llimit, rlimit+1))
    samples, labels = [], []

    for i in range(n):
        eta = np.random.choice(etas, 1)[0]
        mean1 = np.random.uniform(0, 1)
        if theta == -1:
            theta = np.random.uniform(0,1)
        mean2 = mean1 + theta * np.random.choice([-1, 1], 1)[0]
        sample = np.concatenate((np.random.normal(mean1, var, eta),
        np.random.normal(mean2, var, T-eta)))
        samples.append(sample)
        labels.append(eta)
    
    return samples, labels

def generate2(T, var, theta, n):
    llimit = int(T * 0.1)
    rlimit = int(T * 0.9)
    etas = np.array(range(llimit, rlimit+1))
    samples, labels = [], []

    for i in range(n):
        eta = np.random.choice(etas, 1)[0]
        # mean1 is either chosen random uniformly on [0,1] or from a discrate set of values
        # mean1 = np.random.uniform(0, 1)
        mean1 = np.random.choice([0,1], 1)[0]
        if theta == -1:
            theta = np.random.uniform(0,1)
        mean2 = mean1 + theta * np.random.choice([-1, 1], 1)[0]
        sample = np.concatenate((np.random.normal(mean1, var, eta),
        np.random.normal(mean2, var, T-eta)))
        samples.append(sample)
        labels.append(eta)
    
    return samples, labels

def generatewrapper(T, var):
    ns = [1500, 10000]
    thetas = [1]
    datasets_dict = {}

    for theta in thetas:
        x_test, y_test = generate(T, var, theta, 100)
        for n in ns:
            name = 'Discrete2_DoubleUniNormal' + '_theta='+str(theta) + '_n='+str(n)
            x_train, y_train = generate(T, var, theta, n)
            datasets_dict[name] = (np.array(x_train), np.array(y_train),
                                    np.array(x_test), np.array(y_test))
    
    return datasets_dict
    
if __name__ == "__main__":
    dir0 = root_dir + 'data'
    dir1 = root_dir + 'data/original/'
    dir2 = root_dir + 'data/csv/'
    dirs = [dir0, dir1, dir2]
    for diri in dirs:
        if not os.path.isdir(diri):
            os.mkdir(diri)
    
    datasets_dict = generatewrapper(100, 0.2)
    for dsname in datasets_dict:
        # save pickle file first
        with open(dir1+dsname+'.pickle', 'wb') as f:
            pickle.dump(datasets_dict[dsname], f)
        # save test data to csv files for R
        pd.DataFrame(datasets_dict[dsname][2]).T.to_csv(dir2+dsname + '_x_test.csv')
        pd.DataFrame(datasets_dict[dsname][3]).T.to_csv(dir2+dsname + '_y_test.csv')
        # save grouped data
        dataset = datasets_dict[dsname]