from scipy.sparse.linalg import cg
import scipy.sparse as sp
import numpy as np
import pprint

num = 30

data = np.zeros((2 * num - 1, num))
diags = np.arange(-num, num-1, 1)
for n in range(num):
    data[n, :] = np.arange(-1, num-1, 1)
data[num, :] = np.arange(1, num+1, 1)
for n in range(num+1, 2 * num-1):
    data[n, :] = np.arange(num-1-n, 2*num-1-n, 1)
A = sp.spdiags(data, diags, num, num)
pprint.pprint(A.toarray())

b = np.sum(A, axis=1)
tol = 0.000001
x = cg(A, b, tol=tol)
pprint.pprint(x)
