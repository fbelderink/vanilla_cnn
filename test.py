import numpy as np

a = np.random.rand(2,2)

print(a)
print(np.rot90(a,2))
print(np.rot90(a[1:2,1:2],1))