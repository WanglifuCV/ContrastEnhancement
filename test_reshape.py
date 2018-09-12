import numpy as np
import pprint


A = np.arange(25).reshape((5, 5))
print 'A = \n'
pprint.pprint(A)

B = A.T.reshape((A.shape[0] * A.shape[1], 1))
print 'B= \n'
pprint.pprint(B)
