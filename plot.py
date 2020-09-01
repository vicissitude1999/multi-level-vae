import numpy as np
import matplotlib.pyplot as plt

'''
log_names = ['log_exp_1.txt', 'log_exp_3_T=200.txt', 'log_exp_3_T=500.txt',
'log_exp_3_T=1000.txt', 'log_exp_3_T=2000.txt']
title_names = ['experiment 1, n = 5, T = 10k',
'experiment 3, n = 100, T = 200',
'experiment 3, n = 100, T = 500',
'experiment 3, n = 100, T = 1000',
'experiment 3, n = 100, T = 2000']

for i in range(len(log_names)):
    data = np.loadtxt(log_names[i], skiprows=1)

    losses = []
    curr = 0
    loss = 0

    for ind in data:
        if ind[0] == curr:
            loss += ind[2] + ind[3] + ind[4]
        else:
            losses.append(loss)
            loss = 0
            curr += 1
    losses.append(loss)

    x = range(len(losses))

    plt.scatter(x, losses, s=0.8)
    plt.xlabel('epochs')
    plt.ylabel('training losses')
    plt.title(title_names[i])
    plt.savefig('loss_curves/' + log_names[i][0: -4] + '.jpg')
    plt.close()
'''

errors = np.loadtxt('errors.txt')
etas = range(50-20, 50+40)
plt.scatter(etas, errors, s=0.9)
# plt.axvline(x=65)
plt.xlabel('etas')
plt.ylabel('squared errors')
plt.title('Approach 2. Squared errors for Experiment 3 T = 200')
plt.show()