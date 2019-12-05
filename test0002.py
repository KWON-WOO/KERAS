import numpy as np

aaa = np.array([1,2,3,4,5])
print(aaa.shape) # (5,) ≒ (1,5)

aaa = aaa.reshape(1,5)
print(aaa.shape)

bbb = np.array([[1,2,3],[4,5,6]])
print(bbb.shape)

ccc = np.array([[1,2],[3,4],[5,6]])
print(ccc.shape)

ddd = ccc.reshape(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6)
print(ddd)