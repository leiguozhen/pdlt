
# 操作 numpy 数组的常用函数

### where

使用 where 函数能将索引掩码转换成索引位置：

```

indices = where(mask)
indices

=> (array([11, 12, 13, 14]),)

x[indices] # this indexing is equivalent to the fancy indexing x[mask]

=> array([ 5.5,  6. ,  6.5,  7. ])


```

### diag

使用 diag 函数能够提取出数组的对角线(主对角线，负对角线)：

```

diag(A)

=> array([ 0, 11, 22, 33, 44])

diag(A, -1)

array([10, 21, 32, 43])

```

### take

take 函数与高级索引（fancy indexing）用法相似：

```

v2 = arange(-3,3)
v2

=> array([-3, -2, -1,  0,  1,  2])

row_indices = [1, 3, 5]
v2[row_indices] # fancy indexing

=> array([-2,  0,  2])

v2.take(row_indices)
=> array([-2,  0,  2])
```

但是 take 也可以用在 list 和其它对象上：

```
take([-3, -2, -1,  0,  1,  2], row_indices)

=> array([-2,  0,  2])

```

### choose

选取多个数组的部分组成新的数组：

```
which = [1, 0, 1, 0]
choices = [[-2,-2,-2,-2], [5,5,5,5]]
choose(which, choices)

=> array([ 5, -2,  5, -2])
```

# 线性代数

矢量化是用 Python/Numpy 编写高效数值计算代码的关键，这意味着在程序中尽量选择使用矩阵或者向量进行运算，比如矩阵乘法等。

### 标量运算

我们可以使用一般的算数运算符，比如加减乘除，对数组进行标量运算。

```
v1 = arange(0, 5)
v1 * 2

=> array([0, 2, 4, 6, 8])

v1 + 2

=> array([2, 3, 4, 5, 6])

A * 2, A + 2

=> (array([[ 0,  2,  4,  6,  8],
           [20, 22, 24, 26, 28],
           [40, 42, 44, 46, 48],
           [60, 62, 64, 66, 68],
           [80, 82, 84, 86, 88]]),
    array([[ 2,  3,  4,  5,  6],
           [12, 13, 14, 15, 16],
           [22, 23, 24, 25, 26],
           [32, 33, 34, 35, 36],
           [42, 43, 44, 45, 46]]))

```

### Element-wise(逐项乘) 数组-数组 运算

当我们在矩阵间进行加减乘除时，它的默认行为是 element-wise(逐项乘) 的:

```
A * A # element-wise multiplication

=> array([[   0,    1,    4,    9,   16],
          [ 100,  121,  144,  169,  196],
          [ 400,  441,  484,  529,  576],
          [ 900,  961, 1024, 1089, 1156],
          [1600, 1681, 1764, 1849, 1936]])

v1 * v1

=> array([ 0,  1,  4,  9, 16])

A.shape, v1.shape

=> ((5, 5), (5,))

A * v1

=> array([[  0,   1,   4,   9,  16],
          [  0,  11,  24,  39,  56],
          [  0,  21,  44,  69,  96],
          [  0,  31,  64,  99, 136],
          [  0,  41,  84, 129, 176]])

```

# 矩阵代数

### 矩阵乘法

- 1.使用 dot 函数进行 矩阵－矩阵，矩阵－向量，数量积乘法：

```

dot(A, A)

=> array([[ 300,  310,  320,  330,  340],
          [1300, 1360, 1420, 1480, 1540],
          [2300, 2410, 2520, 2630, 2740],
          [3300, 3460, 3620, 3780, 3940],
          [4300, 4510, 4720, 4930, 5140]])

dot(A, v1)

=> array([ 30, 130, 230, 330, 430])

dot(v1, v1)

=> 30

```

- 2.将数组对象映射到 matrix 类型。

```

M = matrix(A)
v = matrix(v1).T # make it a column vector
v

=> matrix([[0],
           [1],
           [2],
           [3],
           [4]])

M * M

=> matrix([[ 300,  310,  320,  330,  340],
           [1300, 1360, 1420, 1480, 1540],
           [2300, 2410, 2520, 2630, 2740],
           [3300, 3460, 3620, 3780, 3940],
           [4300, 4510, 4720, 4930, 5140]])

M * v

=> matrix([[ 30],
           [130],
           [230],
           [330],
           [430]])

# inner product
v.T * v

=> matrix([[30]])

# with matrix objects, standard matrix algebra applies
v + M*v

=> matrix([[ 30],
           [131],
           [232],
           [333],
           [434]])

加减乘除不兼容的维度时会报错：

v = matrix([1,2,3,4,5,6]).T
shape(M), shape(v)

=> ((5, 5), (6, 1))

M * v

  => Traceback (most recent call last):

       File "<ipython-input-9-995fb48ad0cc>", line 1, in <module>
         M * v

       File "/Applications/Spyder-Py2.app/Contents/Resources/lib/python2.7/numpy/matrixlib/defmatrix.py", line 341, in __mul__
         return N.dot(self, asmatrix(other))

     ValueError: shapes (5,5) and (6,1) not aligned: 5 (dim 1) != 6 (dim 0)

```

查看其它运算函数: inner, outer, cross, kron, tensordot。 可以使用 help(kron)。



### 数组/矩阵 变换

之前我们使用 .T 对 v 进行了转置。 我们也可以使用 transpose 函数完成同样的事情。

让我们看看其它变换函数：

```
C = matrix([[1j, 2j], [3j, 4j]])
C

=> matrix([[ 0.+1.j,  0.+2.j],
           [ 0.+3.j,  0.+4.j]])
```

- 共轭：

```
conjugate(C)

=> matrix([[ 0.-1.j,  0.-2.j],
           [ 0.-3.j,  0.-4.j]])

```

- 共轭转置：
 
```
C.H

=> matrix([[ 0.-1.j,  0.-3.j],
           [ 0.-2.j,  0.-4.j]])
```

- real 与 imag 能够分别得到复数的实部与虚部：

```
real(C) # same as: C.real

=> matrix([[ 0.,  0.],
           [ 0.,  0.]])

imag(C) # same as: C.imag

=> matrix([[ 1.,  2.],
           [ 3.,  4.]])
```

- angle 与 abs 可以分别得到幅角和绝对值：

```
angle(C+1) # heads up MATLAB Users, angle is used instead of arg

=> array([[ 0.78539816,  1.10714872],
          [ 1.24904577,  1.32581766]])

abs(C)

=> matrix([[ 1.,  2.],
           [ 3.,  4.]])
```

### 矩阵计算

- 矩阵求逆

```

from scipy.linalg import *
inv(C) # equivalent to C.I 

=> matrix([[ 0.+2.j ,  0.-1.j ],
           [ 0.-1.5j,  0.+0.5j]])




C.I * C

=> matrix([[  1.00000000e+00+0.j,   4.44089210e-16+0.j],
           [  0.00000000e+00+0.j,   1.00000000e+00+0.j]])
```

- 行列式

```

linalg.det(C)

=> (2.0000000000000004+0j)

linalg.det(C.I)

=> (0.50000000000000011+0j)

```

# 数据处理

将数据集存储在 Numpy 数组中能很方便地得到统计数据。为了有个感性地认识，让我们用 numpy 来处理斯德哥尔摩天气的数据。

```

# reminder, the tempeature dataset is stored in the data variable:
shape(data)

=> (77431, 7)
```

平均值

```
# the temperature data is in column 3
mean(data[:,3])

=> 6.1971096847515925
过去200年里斯德哥尔摩的日均温度大约是 6.2 C。
```

标准差 与 方差
```
std(data[:,3]), var(data[:,3])

=> (8.2822716213405663, 68.596023209663286)
```
最小值 与 最大值
```
# lowest daily average temperature
data[:,3].min()

=> -25.800000000000001

# highest daily average temperature
data[:,3].max()

=> 28.300000000000001
```
总和, 总乘积 与 对角线和
```
d = arange(0, 10)
d

=> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# sum up all elements
sum(d)

=> 45

# product of all elements
prod(d+1)

=> 3628800

# cummulative sum
cumsum(d)

=> array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45])

# cummulative product
cumprod(d+1)

=> array([      1,       2,       6,      24,     120,     720,    5040,
            40320,  362880, 3628800])

# same as: diag(A).sum()
trace(A)

=> 110

```
# 对子数组的操作

我们能够通过在数组中使用索引，高级索引，和其它从数组提取数据的方法来对数据集的子集进行操作。

举个例子，我们会再次用到温度数据集：
```
!head -n 3 stockholm_td_adj.dat

1800  1  1    -6.1    -6.1    -6.1 1
1800  1  2   -15.4   -15.4   -15.4 1
1800  1  3   -15.0   -15.0   -15.0 1
```

该数据集的格式是：年，月，日，日均温度，最低温度，最高温度，地点。

如果我们只是关注一个特定月份的平均温度，比如说2月份，那么我们可以创建一个索引掩码，只选取出我们需要的数据进行操作：

```
unique(data[:,1]) # the month column takes values from 1 to 12

=> array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
        12.])

mask_feb = data[:,1] == 2

# the temperature data is in column 3
mean(data[mask_feb,3])

=> -3.2121095707366085

```

拥有了这些工具我们就拥有了非常强大的数据处理能力。 像是计算每个月的平均温度只需要几行代码：

```

months = arange(1,13)
monthly_mean = [mean(data[data[:,1] == month, 3]) for month in months]

fig, ax = subplots()
ax.bar(months, monthly_mean)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly avg. temp.");
```

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1077timestamp1468326454973.png/wm)

# 对高维数组的操作

###  axis

当诸如 min, max 等函数对高维数组进行操作时，有时我们希望是对整个数组进行该操作，有时则希望是对每一行进行该操作。使用 axis 参数我们可以指定函数的行为：

```
m = rand(3,3)
m

=> array([[ 0.09260423,  0.73349712,  0.43306604],
          [ 0.65890098,  0.4972126 ,  0.83049668],
          [ 0.80428551,  0.0817173 ,  0.57833117]])

# global max
m.max()

=> 0.83049668273782951

# max in each column
m.max(axis=0)

=> array([ 0.80428551,  0.73349712,  0.83049668])

# max in each row
m.max(axis=1)

=> array([ 0.73349712,  0.83049668,  0.80428551])

```

### 改变形状与大小

Numpy 数组的维度可以在底层数据不用复制的情况下进行修改，所以 reshape 操作的速度非常快，即使是操作大数组。

```
A

=> array([[ 0,  1,  2,  3,  4],
          [10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34],
          [40, 41, 42, 43, 44]])

n, m = A.shape
B = A.reshape((1,n*m))
B

=> array([[ 0,  1,  2,  3,  4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
           32, 33, 34, 40, 41, 42, 43, 44]])

B[0,0:5] = 5 # modify the array    
B

=> array([[ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
           32, 33, 34, 40, 41, 42, 43, 44]])

A # and the original variable is also changed. B is only a different view of the same data

=> array([[ 5,  5,  5,  5,  5],
          [10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34],
          [40, 41, 42, 43, 44]])
```


### 我们也可以使用 flatten 函数创建一个高阶数组的向量版本，但是它会将数据做一份拷贝。

```

B = A.flatten()
B

=> array([ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
          32, 33, 34, 40, 41, 42, 43, 44])

B[0:5] = 10    
B

=> array([10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
          32, 33, 34, 40, 41, 42, 43, 44])

A # now A has not changed, because B's data is a copy of A's, not refering to the same data

=> array([[ 5,  5,  5,  5,  5],
          [10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34],
          [40, 41, 42, 43, 44]])
```

### 增加一个新维度: newaxis

newaxis 可以帮助我们为数组增加一个新维度，比如说，将一个向量转换成列矩阵和行矩阵：

```

v = array([1,2,3])
shape(v)

=> (3,)

# make a column matrix of the vector v
v[:, newaxis]

=> array([[1],
          [2],
          [3]])

# column matrix
v[:,newaxis].shape

=> (3, 1)

# row matrix
v[newaxis,:].shape

=> (1, 3)

```

# 叠加与重复数组

函数 repeat, tile, vstack, hstack, 与 concatenate能帮助我们以已有的矩阵为基础创建规模更大的矩阵。

### tile 与 repeat

```
a = array([[1, 2], [3, 4]])
# repeat each element 3 times
repeat(a, 3)

=> array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

# tile the matrix 3 times 
tile(a, 3)

=> array([[1, 2, 1, 2, 1, 2],
          [3, 4, 3, 4, 3, 4]])
concatenate

b = array([[5, 6]])
concatenate((a, b), axis=0)

=> array([[1, 2],
          [3, 4],
          [5, 6]])

concatenate((a, b.T), axis=1)

=> array([[1, 2, 5],
          [3, 4, 6]])

```
### hstack 与 vstack

```
vstack((a,b))

=> array([[1, 2],
          [3, 4],
          [5, 6]])

hstack((a,b.T))

=> array([[1, 2, 5],
          [3, 4, 6]])

```

# 浅拷贝与深拷贝

为了获得高性能，Python 中的赋值常常不拷贝底层对象，这被称作浅拷贝。

```
A = array([[1, 2], [3, 4]])    
A

=> array([[1, 2],
          [3, 4]])

# now B is referring to the same array data as A 
B = A 
# changing B affects A
B[0,0] = 10

B

=> array([[10,  2],
          [ 3,  4]])

A

=> array([[10,  2],
          [ 3,  4]])

```

如果我们希望避免改变原数组数据的这种情况，那么我们需要使用 copy 函数进行深拷贝：

```

B = copy(A)
# now, if we modify B, A is not affected
B[0,0] = -5

B

=> array([[-5,  2],
          [ 3,  4]])

A

=> array([[10,  2],
          [ 3,  4]])

```

# 遍历数组元素

通常情况下，我们是希望尽可能避免遍历数组元素的。因为迭代相比向量运算要慢的多。

但是有些时候迭代又是不可避免的，这种情况下用 Python 的 for 是最方便的：

```

v = array([1,2,3,4])

for element in v:
    print(element)

=> 1
   2
   3
   4



M = array([[1,2], [3,4]])

for row in M:
    print("row", row)

    for element in row:
        print(element)

=> row [1 2]
   1
   2
   row [3 4]
   3
   4
```


当我们需要遍历数组并且更改元素内容的时候，可以使用 enumerate 函数同时获取元素与对应的序号：

```

for row_idx, row in enumerate(M):
    print("row_idx", row_idx, "row", row)

    for col_idx, element in enumerate(row):
        print("col_idx", col_idx, "element", element)

        # update the matrix M: square each element
        M[row_idx, col_idx] = element ** 2

row_idx 0 row [1 2]
col_idx 0 element 1
col_idx 1 element 2
row_idx 1 row [3 4]
col_idx 0 element 3
col_idx 1 element 4



# each element in M is now squared
M




array([[ 1,  4],
       [ 9, 16]])

```

# 矢量化函数

像之前提到的，为了获得更好的性能我们最好尽可能避免遍历我们的向量和矩阵，有时可以用矢量算法代替。首先要做的就是将标量算法转换为矢量算法：
```
def Theta(x):
    """
    Scalar implemenation of the Heaviside step function.
    """
    if x >= 0:
        return 1
    else:
        return 0


Theta(array([-3,-2,-1,0,1,2,3]))


=> Traceback (most recent call last):

     File "<ipython-input-11-1f7d89baf696>", line 1, in <module>
       Theta(array([-3, -2, -1, 0, 1, 2, 3]))

     File "<ipython-input-10-fbb0379ab8cb>", line 2, in Theta
       if x >= 0:

   ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

很显然 Theta 函数不是矢量函数所以无法处理向量。

为了得到 Theta 函数的矢量化版本我们可以使用 vectorize 函数：

```
Theta_vec = vectorize(Theta)
Theta_vec(array([-3,-2,-1,0,1,2,3]))

=> array([0, 0, 0, 1, 1, 1, 1])

```

我们也可以自己实现矢量函数:

```
def Theta(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return 1 * (x >= 0)

Theta(array([-3,-2,-1,0,1,2,3]))

=> array([0, 0, 0, 1, 1, 1, 1])

# still works for scalars as well
Theta(-1.2), Theta(2.6)

=> (0, 1)


```

# 数组与条件判断

```
M

=> array([[ 1,  4],
          [ 9, 16]])

if (M > 5).any():
    print("at least one element in M is larger than 5")
else:
    print("no element in M is larger than 5")

=> at least one element in M is larger than 5

if (M > 5).all():
    print("all elements in M are larger than 5")
else:
    print("all elements in M are not larger than 5")

=> all elements in M are not larger than 5

```

# 类型转换

既然 Numpy 数组是静态类型，数组一旦生成类型就无法改变。但是我们可以显示地对某些元素数据类型进行转换生成新的数组，使用 astype 函数（可查看功能相似的 asarray 函数）：
```
M.dtype

=> dtype('int64')

M2 = M.astype(float)    
M2

=> array([[  1.,   4.],
         [  9.,  16.]])

M2.dtype

=> dtype('float64')

M3 = M.astype(bool)
M3

=> array([[ True,  True],
          [ True,  True]], dtype=bool)

```
# 延伸阅读

http://numpy.scipy.org
http://scipy.org/Tentative_NumPy_Tutorial
http://scipy.org/NumPy_for_Matlab_Users - MATLAB 用户的 Numpy 教程。
License

- 本作品在 知识共享许可协议3.0 下许可授权。
