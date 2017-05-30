
# 特定函数


在计算科学问题时，常常会用到很多特定的函数，SciPy 提供了一个非常广泛的特定函数集合。
[函数列表可参考](http://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special)


为了演示特定函数的一般用法我们拿贝塞尔函数举例：

```
#
# The scipy.special module includes a large number of Bessel-functions
# Here we will use the functions jn and yn, which are the Bessel functions 
# of the first and second kind and real-valued order. We also include the 
# function jn_zeros and yn_zeros that gives the zeroes of the functions jn
# and yn.
#
%matplotlib qt
from scipy.special import jn, yn, jn_zeros, yn_zeros
import matplotlib.pyplot as plt

n = 0    # order
x = 0.0

# Bessel function of first kind
print "J_%d(%f) = %f" % (n, x, jn(n, x))

x = 1.0
# Bessel function of second kind
print "Y_%d(%f) = %f" % (n, x, yn(n, x))

=> J_0(0.000000) = 1.000000
   Y_0(1.000000) = 0.088257



x = linspace(0, 10, 100)

fig, ax = plt.subplots()
for n in range(4):
    ax.plot(x, jn(n, x), label=r"$J_%d(x)$" % n)
ax.legend();

fig
```

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468326551535.png/wm)

```
# zeros of Bessel functions
n = 0 # order
m = 4 # number of roots to compute
jn_zeros(n, m)

=> array([  2.40482556,   5.52007811,   8.65372791,  11.79153444])
```

# 积分

### 数值积分: 求积

此处输入图片的描述 被称作 数值求积，Scipy提供了一些列不同类型的求积函数，像是 quad, dblquad 还有 tplquad 分别对应单积分，双重积分，三重积分。
```
from scipy.integrate import quad, dblquad, tplquad
```
quad 函数有许多参数选项来调整该函数的行为（详情见help(quad)）。

一般用法如下:
```
# define a simple function for the integrand
def f(x):
    return x


x_lower = 0 # the lower limit of x
x_upper = 1 # the upper limit of x

val, abserr = quad(f, x_lower, x_upper)

print "integral value =", val, ", absolute error =", abserr 

=> integral value = 0.5 , absolute error = 5.55111512313e-15
如果我们需要传递额外的参数，可以使用 args 关键字：

def integrand(x, n):
    """
    Bessel function of first kind and order n. 
    """
    return jn(n, x)


x_lower = 0  # the lower limit of x
x_upper = 10 # the upper limit of x

val, abserr = quad(integrand, x_lower, x_upper, args=(3,))

print val, abserr 

=> 0.736675137081 9.38925687719e-13
```
对于简单的函数我们可以直接使用匿名函数：
```
val, abserr = quad(lambda x: exp(-x ** 2), -Inf, Inf)

print "numerical  =", val, abserr

analytical = sqrt(pi)
print "analytical =", analytical

=> numerical  = 1.77245385091 1.42026367809e-08
   analytical = 1.77245385091
```
如例子所示，'Inf' 与 '-Inf' 可以表示数值极限。


### 高阶积分用法类似:

```
def integrand(x, y):
    return exp(-x**2-y**2)

x_lower = 0  
x_upper = 10
y_lower = 0
y_upper = 10

val, abserr = dblquad(integrand, x_lower, x_upper, lambda x : y_lower, lambda x: y_upper)

print val, abserr 

=> 0.785398163397 1.63822994214e-13
```
注意到我们为y积分的边界传参的方式，这样写是因为y可能是关于x的函数。


# 常微分方程 (ODEs)

SciPy 提供了两种方式来求解常微分方程：基于函数 odeint 的API与基于 ode 类的面相对象的API。通常 odeint 更好上手一些，而 ode 类更灵活一些。

这里我们将使用 odeint 函数，首先让我们载入它：
```
from scipy.integrate import odeint, ode
```
常微分方程组的标准形式如下:

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468326628451.png/wm)

当

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468326664124.png/wm)


为了求解常微分方程我们需要知道方程 此处输入图片的描述 与初始条件此处输入图片的描述 注意到高阶常微分方程常常写成引入新的变量作为中间导数的形式。 一旦我们定义了函数 f 与数组 y_0 我们可以使用 odeint 函数：
```
y_t = odeint(f, y_0, t)
```

我们将会在下面的例子中看到 Python 代码是如何实现 f 与 y_0 。


示例：阻尼谐震子

常微分方程问题在计算物理学中非常重要，所以我们接下来要看另一个例子：阻尼谐震子。wiki地址：http://en.wikipedia.org/wiki/Damping

阻尼震子的运动公式：

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468326990932.png/wm)

其中 此处输入图片的描述 是震子的位置, 此处输入图片的描述 是频率,  此处输入图片的描述 是阻尼系数. 为了写二阶标准行事的 ODE 我们引入变量：

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327094838.png/wm)

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327109376.png/wm)

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327123489.png/wm)


在这个例子的实现中，我们会加上额外的参数到 RHS 方程中：


```
def dy(y, t, zeta, w0):
    """
    The right-hand side of the damped oscillator ODE
    """
    x, p = y[0], y[1]

    dx = p
    dp = -2 * zeta * w0 * p - w0**2 * x

    return [dx, dp]


# initial state: 
y0 = [1.0, 0.0]


# time coodinate to solve the ODE for
t = linspace(0, 10, 1000)
w0 = 2*pi*1.0


# solve the ODE problem for three different values of the damping ratio

y1 = odeint(dy, y0, t, args=(0.0, w0)) # undamped
y2 = odeint(dy, y0, t, args=(0.2, w0)) # under damped
y3 = odeint(dy, y0, t, args=(1.0, w0)) # critial damping
y4 = odeint(dy, y0, t, args=(5.0, w0)) # over damped


fig, ax = plt.subplots()
ax.plot(t, y1[:,0], 'k', label="undamped", linewidth=0.25)
ax.plot(t, y2[:,0], 'r', label="under damped")
ax.plot(t, y3[:,0], 'b', label=r"critical damping")
ax.plot(t, y4[:,0], 'g', label="over damped")
ax.legend();

fig
```
![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327143049.png/wm)


# 傅立叶变换

傅立叶变换是计算物理学所用到的通用工具之一。Scipy 提供了使用 NetLib FFTPACK 库的接口，它是用FORTRAN写的。Scipy 还另外提供了很多便捷的函数。不过大致上接口都与 NetLib 的接口差不多。

让我们加载它：
```
from scipy.fftpack import *
```
下面演示快速傅立叶变换，例子使用上节阻尼谐震子的例子：
```
N = len(t)
dt = t[1]-t[0]

# calculate the fast fourier transform
# y2 is the solution to the under-damped oscillator from the previous section
F = fft(y2[:,0]) 

# calculate the frequencies for the components in F
w = fftfreq(N, dt)


fig, ax = plt.subplots(figsize=(9,3))
ax.plot(w, abs(F));

fig
```
![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327156443.png/wm)


既然信号是实数，同时频谱是对称的。那么我们只需要画出正频率所对应部分的图：
```
indices = where(w > 0) # select only indices for elements that corresponds to positive frequencies
w_pos = w[indices]
F_pos = F[indices]


fig, ax = subplots(figsize=(9,3))
ax.plot(w_pos, abs(F_pos))
ax.set_xlim(0, 5);

fig
```

![ima](https://dn-anything-about-doc.qbox.me/document-uid8834labid1078timestamp1468327170900.png/wm)

正如预期的那样，我们可以看到频谱的峰值在1处。1就是我们在上节例子中所选的频率。
