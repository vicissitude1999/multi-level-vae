import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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

'''
etas0 = [339, 281, 198, 328, 251, 258, 297, 206, 154, 142, 356, 209, 160, 209, 290, 373, 374, 188, 368, 173, 375, 273, 289, 266, 169, 236, 310, 184, 312, 225, 134, 309, 335, 280, 269, 360, 238, 211, 334, 190, 299, 331, 189, 135, 320, 153, 221, 245, 142, 225]
etas_hat0 = [338, 280, 201, 328, 251, 259, 295, 207, 155, 143, 358, 209, 159, 209, 289, 373, 374, 188, 368, 173, 375, 274, 289, 264, 169, 236, 310, 184, 312, 225, 134, 309, 335, 280, 268, 358, 238, 211, 333, 190, 299, 334, 191, 135, 320, 154, 221, 250, 142, 225]

etas1 = [96, 72, 56, 84, 117, 63, 109, 83, 88, 72, 105, 117, 106, 75, 109, 86, 95, 114, 85, 51, 94, 71, 98, 71, 118, 128, 101, 102, 119, 61, 85, 54, 143, 52, 114, 103, 127, 125, 104, 119, 104, 84, 115, 77, 57, 99, 115, 50, 144, 61, 91, 73, 115, 65, 150, 142, 89, 131, 74, 69, 101, 84, 150, 52, 148, 85, 56, 88, 76, 67, 93, 54, 62, 74, 74, 109, 119, 137, 107, 104, 146, 136, 123, 133, 65, 103, 60, 94, 106, 95, 102, 107, 116, 61, 54, 132, 134, 71, 120, 50]
etas_hat1 = [96, 72, 55, 84, 117, 63, 110, 83, 87, 81, 105, 117, 106, 75, 109, 77, 75, 114, 85, 50, 95, 71, 105, 71, 118, 128, 101, 102, 119, 61, 85, 54, 143, 52, 114, 103, 128, 124, 107, 119, 103, 81, 116, 77, 57, 113, 115, 52, 144, 61, 91, 73, 115, 65, 150, 133, 89, 130, 74, 69, 101, 84, 150, 52, 148, 85, 56, 88, 76, 67, 93, 54, 62, 74, 74, 109, 119, 137, 107, 100, 146, 136, 121, 134, 65, 105, 60, 96, 106, 95, 103, 108, 116, 60, 54, 132, 135, 71, 119, 50]

etas2 = [14, 26, 27, 16, 28, 32, 12, 34, 20, 37, 12, 18, 34, 27, 25, 30, 29, 23, 28, 35, 21, 12, 37, 37, 32, 35, 36, 27, 37, 23, 35, 24, 17, 13, 16, 28, 34, 19, 20, 36, 32, 33, 17, 19, 20, 27, 35, 33, 29, 28, 14, 30, 32, 17, 29, 18, 31, 16, 35, 13, 30, 25, 37, 36, 25, 35, 17, 16, 31, 12, 36, 29, 18, 22, 24, 35, 35, 14, 18, 15, 34, 13, 30, 18, 20, 15, 28, 22, 34, 14, 31, 23, 15, 21, 35, 32, 15, 35, 16, 19]
etas_hat2 = [15, 27, 47, 17, 32, 32, 11, 33, 40, 36, 13, 16, 30, 37, 25, 30, 29, 18, 27, 32, 17, 12, 35, 38, 29, 35, 36, 26, 37, 37, 35, 27, 13, 13, 20, 29, 34, 34, 22, 37, 32, 33, 17, 14, 20, 27, 35, 33, 33, 28, 12, 30, 31, 10, 37, 19, 30, 11, 34, 13, 30, 19, 34, 37, 25, 47, 17, 14, 31, 28, 36, 29, 11, 21, 30, 35, 24, 10, 20, 14, 46, 16, 30, 13, 20, 12, 27, 23, 34, 14, 31, 30, 13, 21, 35, 32, 15, 35, 16, 20]

etas3 = [15, 11, 13, 8, 12, 8, 6, 11, 14, 14, 8, 9, 14, 10, 8, 13, 6, 5, 9, 5, 14, 7, 11, 10, 9, 5, 13, 11, 5, 14, 5, 11, 15, 13, 7, 10, 9, 9, 13, 8, 15, 7, 7, 11, 10, 15, 9, 10, 6, 9, 10, 13, 11, 15, 8, 6, 15, 12, 15, 14, 6, 8, 11, 8, 14, 7, 6, 5, 13, 15, 8, 8, 10, 5, 5, 6, 5, 15, 14, 5, 15, 6, 8, 5, 14, 11, 7, 11, 13, 14, 15, 10, 10, 13, 10, 11, 10, 7, 10, 10]
etas_hat3 = [16, 11, 13, 9, 12, 8, 6, 10, 15, 5, 11, 10, 15, 13, 8, 16, 2, 5, 9, 13, 6, 6, 11, 9, 9, 3, 13, 8, 5, 15, 10, 8, 2, 13, 7, 16, 9, 6, 9, 8, 9, 5, 7, 13, 10, 13, 9, 4, 4, 6, 10, 8, 10, 15, 12, 6, 14, 15, 15, 1, 2, 11, 11, 8, 12, 12, 5, 5, 5, 16, 8, 13, 5, 5, 5, 7, 16, 15, 12, 5, 15, 12, 2, 1, 12, 11, 12, 9, 13, 14, 9, 10, 10, 11, 11, 16, 10, 12, 10, 13]

etas4 = [6, 6, 6, 3, 3, 6, 2, 4, 4, 6, 6, 7, 3, 2, 2, 5, 4, 5, 7, 5, 7, 5, 4, 2, 6, 4, 5, 5, 2, 4, 2, 4, 4, 3, 6, 6, 3, 5, 5, 4, 3, 6, 6, 7, 6, 4, 7, 6, 2, 7, 2, 6, 4, 7, 4, 2, 4, 2, 4, 6, 4, 6, 2, 5, 7, 5, 4, 4, 4, 6, 4, 4, 5, 2, 2, 4, 3, 4, 3, 7, 3, 4, 3, 7, 2, 3, 4, 5, 3, 6, 3, 6, 3, 2, 4, 7, 5, 5, 6, 2]
etas_hat4 = [6, 3, 6, 5, 9, 7, 3, 5, 3, 9, 9, 8, 4, 6, 2, 4, 1, 4, 7, 5, 7, 3, 6, 4, 4, 4, 5, 3, 3, 4, 2, 6, 4, 7, 6, 3, 8, 4, 7, 3, 2, 6, 6, 8, 5, 4, 6, 7, 2, 7, 2, 8, 4, 7, 8, 1, 3, 4, 4, 8, 4, 6, 2, 3, 9, 6, 4, 3, 5, 3, 4, 4, 3, 7, 3, 7, 7, 2, 3, 1, 3, 4, 3, 7, 9, 7, 3, 1, 3, 7, 1, 6, 5, 6, 4, 2, 8, 6, 6, 2]

etas5 = [2, 2, 2, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 3, 3, 1, 3, 3, 2, 1, 3, 1, 3, 2, 1, 1, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 1, 1, 3, 2, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2]
etas_hat5 = [3, 2, 1, 2, 4, 2, 4, 2, 2, 2, 3, 2, 3, 1, 3, 1, 1, 4, 3, 1, 4, 1, 1, 4, 4, 3, 4, 2, 4, 1, 4, 4, 1, 1, 2, 4, 2, 3, 2, 2, 1, 2, 2, 2, 4, 3, 3, 1, 3, 2, 4, 4, 1, 4, 4, 2, 3, 4, 1, 1, 3, 1, 3, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 3, 1, 2, 4, 4, 2, 3, 2, 4, 4, 2, 3, 3, 1, 4, 4, 2, 1, 3, 2, 1, 2, 3, 1, 4, 4, 2]

avg0 = np.sum(np.abs(np.asarray(etas0) - np.asarray(etas_hat0))) / len(etas0)
avg1 = np.sum(np.abs(np.asarray(etas1) - np.asarray(etas_hat1))) / len(etas1)
avg2 = np.sum(np.abs(np.asarray(etas2) - np.asarray(etas_hat2))) / len(etas2)
avg3 = np.sum(np.abs(np.asarray(etas3) - np.asarray(etas_hat3))) / len(etas3)
avg4 = np.sum(np.abs(np.asarray(etas4) - np.asarray(etas_hat4))) / len(etas4)
avg5 = np.sum(np.abs(np.asarray(etas5) - np.asarray(etas_hat5))) / len(etas5)

T = [500, 200, 50, 20, 10, 5]
avgs = [avg0, avg1, avg2, avg3, avg4, avg5]
anno = [str(avg0/5)+'%', str(avg1/2)+'%', str(avg2*2)+'%', str(avg3*5)+'%',
str(avg4*10)+'%', str(avg5*20)+'%']

plt.scatter(T, avgs)
plt.xlabel('T')
plt.ylabel('|eta - eta_hat|')
plt.title('average L1 errors (n = 100, T=500,200,50,20,10,5)')

for i, txt in enumerate(anno):
    plt.annotate(txt, (T[i], avgs[i]))
    
plt.show()
plt.close()
'''

n = [100, 200, 500, 1000]
anno = ['n=100', 'n=200', 'n=500', 'n=1000']
means = []
stds = []

names = {14: 'n=100_T=500',
15: 'n=200_T=500',
16: 'n=500_T=500',
17: 'n=1000_T=500'}

cps = []

for d in names:
    with open('sqerrors/run' + str(d) + '/' + names[d] + '.txt') as f:
        f.readline()
        cps = f.readline().split(' ')[0:-1]
        cps = np.array([int(i) for i in cps])
        break

diffs = []
means = []

for d in names:
    with open('sqerrors/run'+str(d)+'/'+names[d]+'.txt') as f:
        cps_hat = f.readline().split(' ')[0:-1]
        f.readline()
        cps_hat = np.array([int(i) for i in cps_hat])
        diff = np.abs(cps_hat - cps)
        
        diffs.append(diff)
        means.append(np.sum(diff) / len(diff))

        
        plt.hist(diff, [0,2,5,10,20], weights=np.ones(len(diff)) / len(diff))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('|eta - eta_hat|')
        plt.ylabel('percentage of test samples')
        plt.title('histogram for ' + names[d])
        plt.show()
        plt.close()
        

# plot the average of |eta - eta_hat| across test samples
plt.scatter(range(len(means)), means)
for i, txt in enumerate(anno):
    plt.annotate(txt, (range(len(means))[i], means[i]))
plt.ylabel('average |eta - eta_hat| across all test samples')
plt.title('n vs average |eta - eta_hat|')
plt.show()
plt.close()

# plot error bars
avg = np.average(diffs, axis=0)
std = np.std(diffs, axis=0)

plt.errorbar(range(len(cps)), avg, yerr=std, fmt='o')
plt.xlabel('test samples X1-X50')
plt.ylabel('average |eta - eta_hat| across different n values')
plt.title('n = 1000,500,200,100 T=500')
plt.show()
plt.close()