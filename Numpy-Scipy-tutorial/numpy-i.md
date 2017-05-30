
# 创建 numpy 数组

初始化numpy数组有多种方式，比如说：

- 使用 Python 列表或元祖
- 使用 arange, linspace 等函数
- 从文件中读取数据
- 列表生成numpy数组

我们使用 numpy.array 来创建数组

```

# a vector: the argument to the array function is a Python list
v = array([1,2,3,4])
v 

=> array([1, 2, 3, 4])
（注：=> 后为控制台输出结果）

# a matrix: the argument to the array function is a nested Python list
M = array([[1, 2], [3, 4]])
M 

=> array([[1, 2], 
          [3, 4]]) 
v 与 M 对象都是 numpy 模块提供的 ndarray 类型

type(v), type(M)

=> (<type 'numpy.ndarray'>,<type 'numpy.ndarray'>)
```

v 与 M 数组的不同之处在于它们的维度。 我们可以通过 ndarray.shape 获得它的维度属性：
```
v.shape

=> (4,)

M.shape

=> (2, 2)
```
数组的元素数量可以通过 ndarray.size 得到：
```
M.size

=> 4
```
同样的，我们可以使用 numpy.shape 与 numpy.size 函数获取对应属性值：
```
shape(M)

=> (2, 2)

size(M)

=> 4
```
到目前为止 numpy.ndarray 看上去与 list 差不多。 为什么不直接使用list呢？

原因有以下几点:

- Python 的 list 是动态类型，可以包含不同类型的元素，所以没有支持诸如点乘等数学函数，因为要为 list 实现这些操作会牺牲性能。
- Numpy 数组是 静态类型 并且 齐次。 元素类型在数组创建的时候就已经确定了。
- Numpy 数组节约内存。
- 由于是静态类型，对其数学操作函数（如矩阵乘法，矩阵加法）的实现可以使用 C 或者 Fortran 完成。


使用 ndarray 的 dtype 属性我们能获得数组元素的类型：
```
M.dtype

=> dtype('int64')
```
当我们试图为一个 numpy 数组赋错误类型的值的时候会报错：
```
M[0,0] = "hello"

=> Traceback (most recent call last):

       File "<ipython-input-4-a09d72434238>", line 1, in <module>
           M[0,0] = "hello"

   ValueError: invalid literal for long() with base 10: 'hello'
```
我们可以显示地定义元素类型通过在创建数组时使用 dtype 关键字参数：
```
M = array([[1, 2], [3, 4]], dtype=complex)    
M

=> array([[ 1.+0.j,  2.+0.j],
          [ 3.+0.j,  4.+0.j]])
```

dtype 的常用值有：int, float, complex, bool, object 等。

我们也可以显示的定义数据类型的大小，比如：int64, int16, float128, complex128.

使用数组生成函数

当需要生产大数组时，手动创建显然是不明智的，我们可以使用函数来生成数组，最常用的有如下几个函数：

arange
```
# create a range
x = arange(0, 10, 1) # arguments: start, stop, step    
x

=> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


x = arange(-1, 1, 0.1)
x

=>array([ -1.00000000e+00,  -9.00000000e-01,  -8.00000000e-01,
          -7.00000000e-01,  -6.00000000e-01,  -5.00000000e-01,
          -4.00000000e-01,  -3.00000000e-01,  -2.00000000e-01,
          -1.00000000e-01,  -2.22044605e-16,   1.00000000e-01,
           2.00000000e-01,   3.00000000e-01,   4.00000000e-01,
           5.00000000e-01,   6.00000000e-01,   7.00000000e-01,
           8.00000000e-01,   9.00000000e-01])
```
linspace 与 logspace
```
# using linspace, both end points ARE included
linspace(0, 10, 25)

=> array([  0.        ,   0.41666667,   0.83333333,   1.25      ,
            1.66666667,   2.08333333,   2.5       ,   2.91666667,
            3.33333333,   3.75      ,   4.16666667,   4.58333333,
            5.        ,   5.41666667,   5.83333333,   6.25      ,
            6.66666667,   7.08333333,   7.5       ,   7.91666667,
            8.33333333,   8.75      ,   9.16666667,   9.58333333,  10.        ])




logspace(0, 10, 10, base=e)

=> array([  1.00000000e+00,   3.03773178e+00,   9.22781435e+00,
            2.80316249e+01,   8.51525577e+01,   2.58670631e+02,
            7.85771994e+02,   2.38696456e+03,   7.25095809e+03,
            2.20264658e+04])
```

mgrid

```
x, y = mgrid[0:5, 0:5] # similar to meshgrid in MATLAB
x

=> array([[0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4]])

y

=> array([[0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4]])
```
random data
```
from numpy import random

# uniform random numbers in [0,1]
random.rand(5,5)

=> array([[ 0.30550798,  0.91803791,  0.93239421,  0.28751598,  0.04860825],
          [ 0.45066196,  0.76661561,  0.52674476,  0.8059357 ,  0.1117966 ],
          [ 0.05369232,  0.48848972,  0.74334693,  0.71935866,  0.35233569],
          [ 0.13872424,  0.58346613,  0.37483754,  0.59727255,  0.38859949],
          [ 0.29037136,  0.8360109 ,  0.63105782,  0.58906755,  0.64758577]])

# standard normal distributed random numbers
random.randn(5,5)

=> array([[ 0.28795069, -0.35938689, -0.31555872,  0.48542156,  0.26751156],
          [ 2.13568908,  0.85288911, -0.70587016,  0.98492216, -0.99610179],
          [ 0.49670578, -0.08179433,  0.58322716, -0.21797477, -1.16777687],
          [-0.3343575 ,  0.20369114, -0.31390896,  0.3598063 ,  0.36981814],
          [ 0.4876012 ,  1.9979494 ,  0.75177876, -1.80697478,  1.64068423]])
```
diag
```
# a diagonal matrix
diag([1,2,3])

=> array([[1, 0, 0],
          [0, 2, 0],
          [0, 0, 3]])

# diagonal with offset from the main diagonal
diag([1,2,3], k=1) 

=> array([[0, 1, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]])
```
zeros 与 ones
```
zeros((3,3))

=> array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]])

ones((3,3))

=> array([[ 1.,  1.,  1.],
          [ 1.,  1.,  1.],
          [ 1.,  1.,  1.]])
```
# 文件 I/O 创建数组

CSV

CSV是一种常用的数据格式化文件类型，为了从中读取数据，我们使用 numpy.genfromtxt 函数。

数据文件 stockholm_td_adj.dat 就在工作目录下，文件格式如下：
```
!head stockholm_td_adj.dat

1800  1  1    -6.1    -6.1    -6.1 1
1800  1  2   -15.4   -15.4   -15.4 1
1800  1  3   -15.0   -15.0   -15.0 1
1800  1  4   -19.3   -19.3   -19.3 1
1800  1  5   -16.8   -16.8   -16.8 1
1800  1  6   -11.4   -11.4   -11.4 1
1800  1  7    -7.6    -7.6    -7.6 1
1800  1  8    -7.1    -7.1    -7.1 1
1800  1  9   -10.1   -10.1   -10.1 1
1800  1 10    -9.5    -9.5    -9.5 1
```
可视化数据（可视化的内容在matplotlib章节中，这里先小小演示下）：
```
%matplotlib inline
import matplotlib.pyplot as plt
data = genfromtxt('stockholm_td_adj.dat')
data.shape

=> (77431, 7)

fig, ax = plt.subplots(figsize=(14,4))

ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('temperatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature (C)')
fig
```


使用 numpy.savetxt 我们可以将 Numpy 数组保存到csv文件中:
```
M = random.rand(3,3)
M

=> array([[ 0.70506801,  0.54618952,  0.31039856],
          [ 0.26640475,  0.10358152,  0.73231132],
          [ 0.07987128,  0.34462854,  0.91114433]])

savetxt("random-matrix.csv", M)

!cat random-matrix.csv

=> 7.050680113576863750e-01 5.461895177867910345e-01 3.103985627238065037e-01
   2.664047486311884594e-01 1.035815249084012235e-01 7.323113219935466489e-01
   7.987128326702574999e-02 3.446285401590922781e-01 9.111443300153220237e-01



savetxt("random-matrix.csv", M, fmt='%.5f') # fmt specifies the format

!cat random-matrix.csv

=> 0.70507 0.54619 0.31040
   0.26640 0.10358 0.73231
   0.07987 0.34463 0.91114
```
Numpy 原生文件类型

使用 numpy.save 与 numpy.load 保存和读取：
```
save("random-matrix.npy", M)

!file random-matrix.npy

=> random-matrix.npy: data

load("random-matrix.npy")

=> array([[ 0.70506801,  0.54618952,  0.31039856],
          [ 0.26640475,  0.10358152,  0.73231132],
          [ 0.07987128,  0.34462854,  0.91114433]])
```
numpy 数组的常用属性
```

M.itemsize # bytes per element

=> 8

M.nbytes # number of bytes

=> 72



M.ndim # number of dimensions

=> 2
```
# 操作数组

索引

最基本的，我们用方括号进行检索：
```
# v is a vector, and has only one dimension, taking one index
v[0]

=> 1

# M is a matrix, or a 2 dimensional array, taking two indices 
M[1,1]

=> 0.10358152490840122
```

如果是N(N > 1)维数列，而我们在检索时省略了一个索引值则会返回一整行((N-1)维数列)：
```
M
=> array([[ 0.70506801,  0.54618952,  0.31039856],
          [ 0.26640475,  0.10358152,  0.73231132],
          [ 0.07987128,  0.34462854,  0.91114433]])

M[1]
=> array([ 0.26640475,  0.10358152,  0.73231132])
```
使用 : 能达到同样的效果:
```
M[1,:] # row 1

=> array([ 0.26640475,  0.10358152,  0.73231132])

M[:,1] # column 1

=> array([ 0.54618952,  0.10358152,  0.34462854])
```
我们可以利用索引进行赋值：
```
M[0,0] = 1
M

=> array([[ 1.        ,  0.54618952,  0.31039856],
          [ 0.26640475,  0.10358152,  0.73231132],
          [ 0.07987128,  0.34462854,  0.91114433]])


# also works for rows and columns
M[1,:] = 0
M[:,2] = -1
M

=> array([[ 1.        ,  0.54618952, -1.        ],
          [ 0.        ,  0.        , -1.        ],
          [ 0.07987128,  0.34462854, -1.        ]])
```
切片索引

切片索引语法：M[lower:upper:step]
```
A = array([1,2,3,4,5])
A

=> array([1, 2, 3, 4, 5])

A[1:3]

=> array([2, 3])
```
进行切片赋值时，原数组会被修改：
```
A[1:3] = [-2,-3]
A

=> array([ 1, -2, -3,  4,  5])
```
我们可以省略 M[lower:upper:step] 中的任意参数:
```
A[::] # lower, upper, step all take the default values

=> array([ 1, -2, -3,  4,  5])

A[::2] # step is 2, lower and upper defaults to the beginning and end of the array

=> array([ 1, -3,  5])

A[:3] # first three elements

=> array([ 1, -2, -3])

A[3:] # elements from index 3

=> array([4, 5])
```
负值索引从数组尾开始计算：
```
A = array([1,2,3,4,5])
A[-1] # the last element in the array

=> 5

A[-3:] # the last three elements

=> array([3, 4, 5])
```
索引切片在多维数组的应用也是一样的:
```
A = array([[n+m*10 for n in range(5)] for m in range(5)])
A

=> array([[ 0,  1,  2,  3,  4],
          [10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34],
          [40, 41, 42, 43, 44]])


# a block from the original array
A[1:4, 1:4]

=> array([[11, 12, 13],
          [21, 22, 23],
          [31, 32, 33]])

# strides
A[::2, ::2]

=> array([[ 0,  2,  4],
          [20, 22, 24],
          [40, 42, 44]])
```
# 高级索引（Fancy indexing）

指使用列表或者数组进行索引:
```
row_indices = [1, 2, 3]
A[row_indices]

=> array([[10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34]])

col_indices = [1, 2, -1] # remember, index -1 means the last element
A[row_indices, col_indices]

=> array([11, 22, 34])
```
我们也可以使用索引掩码:
```
B = array([n for n in range(5)])
B

=> array([0, 1, 2, 3, 4])

row_mask = array([True, False, True, False, False])
B[row_mask]

=> array([0, 2])


# same thing
row_mask = array([1,0,1,0,0], dtype=bool)
B[row_mask]

=> array([0, 2])
```
使用比较操作符生成掩码:
```
x = arange(0, 10, 0.5)
x

=> array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,
           5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5])

mask = (5 < x) * (x < 7.5)
mask

=> array([False, False, False, False, False, False, False, False, False,
          False, False,  True,  True,  True,  True, False, False, False,
          False, False], dtype=bool)

x[mask]

=> array([ 5.5,  6. ,  6.5,  7. ])
```
