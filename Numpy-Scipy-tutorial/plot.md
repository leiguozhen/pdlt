
# 类MATLAB API

最简单的入门是从类 MATLAB API 开始，它被设计成兼容 MATLAB 绘图函数。

让我们加载它：
```
from pylab import *
```

使用 qt 作为图形后端：

```
%matplotlib qt
```
示例

类MATLAB API 绘图的简单例子:
```
from numpy import *
x = linspace(0, 5, 10)
y = x ** 2

figure()
plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('title')
show()
```
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1080timestamp1468327804435.png/wm)


创建子图，选择绘图用的颜色与描点符号:
```
subplot(1,2,1)
plot(x, y, 'r--')
subplot(1,2,2)
plot(y, x, 'g*-');
```
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1080timestamp1468327845820.png/wm)

此类 API 的好处是可以节省你的代码量，但是我们并不鼓励使用它处理复杂的图表。

# [延伸阅读](https://www.shiyanlou.com/courses/348)

