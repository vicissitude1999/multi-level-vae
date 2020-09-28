import numpy as np
import pandas as pd
import pickle
import os

# fixed variance, mean generated differently
# mean1 is tuple or list
# mean2 is int or list
def generate(T, var, theta, n, m1=(0,1), m2=1):
    llimit = int(T * 0.1)
    rlimit = int(T * 0.9)
    etas = np.array(range(llimit, rlimit+1))
    samples, labels = [], []

    for i in range(n):
        eta = np.random.choice(etas, 1)[0]
        if isinstance(m1, tuple):
            mean1 = np.random.uniform(m1[0], m1[1])
        elif isinstance(m1, list):
            mean1 = np.random.choice(m1, 1)[0]
        else:
            print('shit')
            raise Exception('invalid mean1 type')

        if theta == -1:
            theta = np.random.uniform(0,1)
        
        if isinstance(m2, int):
            mean2 = mean1 + theta * m2 * np.random.choice([1,-1], 1)[0]
        elif isinstance(m2, list):
            mean2 = np.random.choice(m2, 1)[0]
        else:
            raise Exception('invalid mean2 type')

        sample = np.concatenate((np.random.normal(mean1, var, eta),
        np.random.normal(mean2, var, T-eta)))
        
        samples.append(sample)
        labels.append(eta)
    
    return samples, labels

def generatewrapper(T, var):
    thetas = [1, 0.75, 0.5, 0.25, -1]
    ns = [1500, 10000]
    mean1s = [(0,1), [0,1,2,3,4,5,6,7,8,9], [0,2,4,6,8]]
    mean2s = [1, 1, [1,3,5,7,9]]
    datasets_dict = {}

    for i in range(len(mean1s)):
        mean1 = mean1s[i]
        mean2 = mean2s[i]
        print(mean1, mean2)

        for theta in thetas:
            # generate test data
            x_test, y_test = generate(T, var, theta, 100, mean1, mean2)
            for n in ns:
                name = 'T={}_var={}_theta={}_n={}_m1={}_m2={}'.format(T,var,theta,n, mean1, mean2)
                x_train, y_train = generate(T, var, theta, n, mean1, mean2)
                datasets_dict[name] = (np.array(x_train), np.array(y_train),
                                        np.array(x_test), np.array(y_test))
            # if mean2 is chosen from a discrete set of values, no need to use theta
            if isinstance(mean2, list):
                break
    
    return datasets_dict
    
if __name__ == "__main__":
    root_dir = os.getcwd()

    dirs = [os.path.join(root_dir, 'data/'), os.path.join(root_dir, 'data/csv/')]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    
    datasets_dict = generatewrapper(100, 0.2)
    for dsname in datasets_dict:
        # save pickle file first
        with open(os.path.join(dirs[0], dsname), 'wb') as f:
            pickle.dump(datasets_dict[dsname], f)
        # save test data to csv files for R
        # pd.DataFrame(datasets_dict[dsname][2]).T.to_csv(dir2+dsname + '_x_test.csv')
        # pd.DataFrame(datasets_dict[dsname][3]).T.to_csv(dir2+dsname + '_y_test.csv')