import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

n_gaussians = 7

df= pd.read_csv('ToClusterMSD.csv')


a = np.genfromtxt('ToClusterMSD.csv', delimiter=',')

q, w = np.where(np.isnan(a))
a[q, w] = 0

gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
gmm.fit(a)

p = gmm.predict(a)
p = p.reshape(-1, 1)
t = np.append(a, p, axis=1)

df = pd.DataFrame(t)

df.to_csv('clusteredMSD.csv', index=False, header=False)
df2 = pd.read_csv('clusteredMSD.csv')
