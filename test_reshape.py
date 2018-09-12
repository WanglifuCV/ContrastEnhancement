import numpy as np
import pprint

data = np.arange(36)
print 'data = \n'
pprint.pprint(data)

A = data.reshape((6, 6), order='C')
print 'A = \n'
pprint.pprint(A)

B = A.T.reshape((4, 9), order='C')
print 'B= \n'
pprint.pprint(B)

C = B.T.reshape((2, 18), order='C')
print 'C= \n'
pprint.pprint(C)

D = A.T.reshape((A.shape[0] * A.shape[1], 1), order='C')
print 'C= \n'
pprint.pprint(D)
