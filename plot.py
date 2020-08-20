import numpy as np
import matplotlib.pyplot as plt
'''
log_names = ['log.txt']

for log_name in log_names:
    data = np.loadtxt(log_name, skiprows=1)

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
    plt.title('MLVAE on double univatiate normal data, n = 1500, T = 100')
    plt.show()
'''

errors = np.loadtxt('errors.txt')
etas = np.array(range(80)) + 10
plt.scatter(etas, errors, s=0.9)
plt.axvline(x=65)
plt.xlabel('etas')
plt.ylabel('squared errors')
plt.title('Approach 2. Squared errors for a single data X_0 in n=1500 theta=1')
plt.show()