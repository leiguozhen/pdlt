
# 线性代数

线性代数模块包含了大量矩阵相关的函数，包括线性方程求解，特征值求解，矩阵函数，分解函数（SVD, LU, cholesky）等等。 详情见：http://docs.scipy.org/doc/scipy/reference/linalg.html

线性方程组

```
from scipy.linalg import *
from numpy.random import *


A = array([[1,2,3], [4,5,6], [7,8,9]])
b = array([1,2,3])

x = solve(A, b)

x

=> array([-0.33333333,  0.66666667,  0.        ])


# check
dot(A, x) - b

=> array([ -1.11022302e-16,   0.00000000e+00,   0.00000000e+00])
```


AXB 都是矩阵，我们也可以这么做:
```
A = rand(3,3)
B = rand(3,3)

X = solve(A, B)

X

=> array([[ 2.28587973,  5.88845235,  1.6750663 ],
          [-4.88205838, -5.26531274, -1.37990347],
          [ 1.75135926, -2.05969998, -0.09859636]])


# check
norm(dot(A, X) - B)

=> 6.2803698347351007e-16
```
特征值 与 特征向量

矩阵 A 的特征值问题:

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1079timestamp1468327348912.png/wm)



使用 eigvals 计算矩阵的特征值，使用 eig 同时计算矩阵的特征值与特征向量：
```
evals = eigvals(A)
evals

=> array([ 1.06633891+0.j        , -0.12420467+0.10106325j,
          -0.12420467-0.10106325j])


evals, evecs = eig(A)
evals

=> array([ 1.06633891+0.j        , -0.12420467+0.10106325j,
          -0.12420467-0.10106325j])

evecs
=> array([[ 0.89677688+0.j        , -0.30219843-0.30724366j, -0.30219843+0.30724366j],
          [ 0.35446145+0.j        ,  0.79483507+0.j        ,  0.79483507+0.j        ],
          [ 0.26485526+0.j        , -0.20767208+0.37334563j, -0.20767208-0.37334563j]])
```
第 n 个特征值(存储在 evals[n])所对应的特征向量是evecs 的第n列, 比如, evecs[:,n]。为了验证这点， 让我们将特征向量乘上矩阵，比较乘积与特征值：
```
n = 1    
norm(dot(A, evecs[:,n]) - evals[n] * evecs[:,n])

=> 1.3964254612015911e-16
```

# 矩阵运算
```
# the matrix inverse
inv(A)

=> array([[-1.38585633,  1.36837431,  6.03633364],
          [ 3.80855289, -4.76960426, -5.2571037 ],
          [ 0.0689213 ,  2.4652602 , -2.5948838 ]])

# determinant
det(A)

=> 0.027341548212627968

# norms of various orders
norm(A, ord=2), norm(A, ord=Inf)

=> (1.1657807164173386, 1.7872032588446576)
```

# 稀疏矩阵

稀疏矩阵对于数值模拟一个大的方程组是很有帮助的。SciPy 对稀疏矩阵有着很好的支持，可以对其进行基本的线性代数运算（比如方程求解，特征值计算等）。

有很多种存储稀疏矩阵的方式。一般常用的有坐标形式（COO），列表嵌套列表的形式（LIL）,压缩列（CSC），压缩行（CSR）等。

每一种形式都有它的优缺点。CSR与CSC在大部分算法下都有着不错的性能，但是它们不够直观，也不容易初始化。所以一般情况下我们都会先在COO活着LIL下进行初始化，再转换成CSC活着CSR形式使用。

当我们创建一个稀疏矩阵的时候，我们需要选择它的存储形式：
```
from scipy.sparse import *

# dense matrix
M = array([[1,0,0,0], [0,3,0,0], [0,1,1,0], [1,0,0,1]]); M

=> array([[1, 0, 0, 0],
          [0, 3, 0, 0],
          [0, 1, 1, 0],
          [1, 0, 0, 1]])

# convert from dense to sparse
A = csr_matrix(M); A

=> <4x4 sparse matrix of type '<type 'numpy.int64'>'
       with 6 stored elements in Compressed Sparse Row format>

# convert from sparse to dense
A.todense()

=> matrix([[1, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 1, 1, 0],
           [1, 0, 0, 1]])
```
创建稀疏矩阵更有效率的方法是：先创建一个空矩阵，再按索引进行填充：

```
A = lil_matrix((4,4)) # empty 4x4 sparse matrix
A[0,0] = 1
A[1,1] = 3
A[2,2] = A[2,1] = 1
A[3,3] = A[3,0] = 1
A

=> <4x4 sparse matrix of type '<type 'numpy.float64'>'
       with 6 stored elements in LInked List format>


A.todense()

matrix([[ 1.,  0.,  0.,  0.],
        [ 0.,  3.,  0.,  0.],
        [ 0.,  1.,  1.,  0.],
        [ 1.,  0.,  0.,  1.]])

```

在两种不同的稀疏矩阵格式间转换：
```
A

=> <4x4 sparse matrix of type '<type 'numpy.float64'>'
       with 6 stored elements in LInked List format>

A = csr_matrix(A); A

=> <4x4 sparse matrix of type '<type 'numpy.float64'>'
       with 6 stored elements in Compressed Sparse Row format>

A = csc_matrix(A); A

=> <4x4 sparse matrix of type '<type 'numpy.float64'>'
       with 6 stored elements in Compressed Sparse Column format>
```

可以像计算稠密矩阵一样计算稀疏矩阵:
```
A.todense()

=> matrix([[ 1.,  0.,  0.,  0.],
           [ 0.,  3.,  0.,  0.],
           [ 0.,  1.,  1.,  0.],
           [ 1.,  0.,  0.,  1.]])

(A * A).todense()

=> matrix([[ 1.,  0.,  0.,  0.],
           [ 0.,  9.,  0.,  0.],
           [ 0.,  4.,  1.,  0.],
           [ 2.,  0.,  0.,  1.]])

dot(A, A).todense()

=> matrix([[ 1.,  0.,  0.,  0.],
           [ 0.,  9.,  0.,  0.],
           [ 0.,  4.,  1.,  0.],
           [ 2.,  0.,  0.,  1.]])

v = array([1,2,3,4])[:,newaxis]; v

=> array([[1],
          [2],
          [3],
          [4]])

# sparse matrix - dense vector multiplication
A * v

=> array([[ 1.],
          [ 6.],
          [ 5.],
          [ 5.]])

# same result with dense matrix - dense vector multiplcation
A.todense() * v

=> matrix([[ 1.],
           [ 6.],
           [ 5.],
           [ 5.]])

```

# 最优化

最优化 (找到函数的最大值或最小值) 问题是数学中比较大的话题, 复杂的函数与变量的增加会使问题变得更加困难。这里我们只看一些简单的例子。

想知道更多，详情见： http://scipy- lectures.github.com/advanced/mathematical_optimization/index.html

让我们首先载入模块：
```
from scipy import optimize
```
找到一个最小值

首先看一下单变量简单函数的最优化解法：
```
def f(x):
    return 4*x**3 + (x-2)**2 + x**4


fig, ax  = subplots()
x = linspace(-5, 3, 100)
ax.plot(x, f(x));
```


可以使用 fmin_bfgs 找到函数的最小值：
```
x_min = optimize.fmin_bfgs(f, -2)
x_min 

=> Optimization terminated successfully.
            Current function value: -3.506641
            Iterations: 6
            Function evaluations: 30
            Gradient evaluations: 10

   array([-2.67298167])

optimize.fmin_bfgs(f, 0.5) 

=> Optimization terminated successfully.
            Current function value: 2.804988
            Iterations: 3
            Function evaluations: 15
            Gradient evaluations: 5

   array([ 0.46961745])
```
也可以使用 brent 或者 fminbound函数，区别就是一点语法和实现所用的算法。
```
optimize.brent(f)

=> 0.46961743402759754

optimize.fminbound(f, -4, 2)

=> -2.6729822917513886

```



# 找到方程的解

为了找到 此处输入图片的描述 方程的根，我们可以使用 fsolve。它需要一个初始的猜测值：
```
omega_c = 3.0
def f(omega):
    # a transcendental equation: resonance frequencies of a low-Q SQUID terminated microwave resonator
    return tan(2*pi*omega) - omega_c/omega


fig, ax  = subplots(figsize=(10,4))
x = linspace(0, 3, 1000)
y = f(x)
mask = where(abs(y) > 50)
x[mask] = y[mask] = NaN # get rid of vertical line when the function flip sign
ax.plot(x, y)
ax.plot([0, 3], [0, 0], 'k')
ax.set_ylim(-5,5);
```

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1079timestamp1468327451910.png/wm)
```

optimize.fsolve(f, 0.1)

=> array([ 0.23743014])

optimize.fsolve(f, 0.6)

=> array([ 0.71286972])

optimize.fsolve(f, 1.1)

=> array([ 1.18990285])
```

## 插值

scipy 插值是很方便的：interp1d 函数以一组X，Y数据为输入数据，返回一个类似于函数的对象，输入任意x值给该对象，返回对应的内插值y：

```
from scipy.interpolate import *


def f(x):
    return sin(x)


n = arange(0, 10)  
x = linspace(0, 9, 100)

y_meas = f(n) + 0.1 * randn(len(n)) # simulate measurement with noise
y_real = f(x)

linear_interpolation = interp1d(n, y_meas)
y_interp1 = linear_interpolation(x)

cubic_interpolation = interp1d(n, y_meas, kind='cubic')
y_interp2 = cubic_interpolation(x)


fig, ax = subplots(figsize=(10,4))
ax.plot(n, y_meas, 'bs', label='noisy data')
ax.plot(x, y_real, 'k', lw=2, label='true function')
ax.plot(x, y_interp1, 'r', label='linear interp')
ax.plot(x, y_interp2, 'g', label='cubic interp')
ax.legend(loc=3);
```
![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1079timestamp1468327631288.png/wm)

# 统计学

scipy.stats 模块包含了大量的统计分布，统计函数与测试。

完整的文档请查看：http://docs.scipy.org/doc/scipy/reference/stats.html

```
from scipy import stats


# create a (discreet) random variable with poissionian distribution

X = stats.poisson(3.5) # photon distribution for a coherent state with n=3.5 photons


n = arange(0,15)

fig, axes = subplots(3,1, sharex=True)

# plot the probability mass function (PMF)
axes[0].step(n, X.pmf(n))

# plot the commulative distribution function (CDF)
axes[1].step(n, X.cdf(n))

# plot histogram of 1000 random realizations of the stochastic variable X
axes[2].hist(X.rvs(size=1000));
```
![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1079timestamp1468327689283.png/wm)
```
# create a (continous) random variable with normal distribution
Y = stats.norm()


x = linspace(-5,5,100)

fig, axes = subplots(3,1, sharex=True)

# plot the probability distribution function (PDF)
axes[0].plot(x, Y.pdf(x))

# plot the commulative distributin function (CDF)
axes[1].plot(x, Y.cdf(x));

# plot histogram of 1000 random realizations of the stochastic variable Y
axes[2].hist(Y.rvs(size=1000), bins=50);
```
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1079timestamp1468327715235.png/wm)

统计值:
```
X.mean(), X.std(), X.var() # poission distribution

=> (3.5, 1.8708286933869707, 3.5)


Y.mean(), Y.std(), Y.var() # normal distribution

=> (0.0, 1.0, 1.0)
```
统计检验

检验两组独立的随机数据是否来组同一个分布：
```
t_statistic, p_value = stats.ttest_ind(X.rvs(size=1000), X.rvs(size=1000))

print "t-statistic =", t_statistic
print "p-value =", p_value

=> t-statistic = -0.244622880865
   p-value = 0.806773564698
```
既然P值很大，我们就不能拒绝两组数据拥有不同的平均值的假设：

检验一组随机数据的平均值是否为 0.1（实际均值为0.1）：
```
stats.ttest_1samp(Y.rvs(size=1000), 0.1)

=> (-4.4661322772225356, 8.8726783620609218e-06)
```
低p值意味着我们可以拒绝y的均值为 0.1 这个假设。
```
Y.mean()

=> 0.0

stats.ttest_1samp(Y.rvs(size=1000), Y.mean())

=> (0.51679431628006112, 0.60541413382728715)
```
# 延伸阅读

http://www.scipy.org - The official web page for the SciPy project.
http://docs.scipy.org/doc/scipy/reference/tutorial/index.html - A tutorial on how to get started using SciPy.
https://github.com/scipy/scipy/ - The SciPy source code.
License

该作品在 知识共享许可协议3.0 下许可授权。
