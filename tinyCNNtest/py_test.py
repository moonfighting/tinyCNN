print '%d/%d' %(3, 4)
import numpy as np


a = np.random.random((4, 3))



print a

print ':,' * 0
for i in range(0, 5):
    return_src = 'a[%s 0,...]' % (':,' * i)
    print return_src

c = eval('a[0, :]')
b = eval('a[:, 0]')
print b