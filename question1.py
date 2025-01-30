import numpy as np
import matplotlib.pyplot as plt
import math
num =  100
sds = [0] * 11
means = [0] * 11
for e in range(11):
    d = 2 ** e
    r_points = np.random.rand(num, d)
    distances = []
    for i in range(100):
        for j in range(i + 1, 100):
            p1 = r_points[i]
            p2 = r_points[j]
            dist = (np.linalg.norm(p1 - p2, ord=1))
            distances.append(dist)
    mean = np.mean(distances)
    std = np.std(distances)
    sds[e] = std
    means[e] = mean
x = [2**e for e in range(11)]
plt.figure(1)  # Create or switch to the first figure
plt.plot(x, means, label='dimension vs mean')
plt.title('Figure 1: L1 distance')
plt.xlabel('dimension')
plt.ylabel('mean')
plt.legend()
plt.figure(2)  # Create or switch to the second figure
plt.plot(x, sds, color='r', label='dimension vs standard deviation')
plt.title('Figure 2: L1 distance')
plt.xlabel('dimension')
plt.ylabel('standard deviation')
plt.legend()
plt.show()


