import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np

# 作函数图像
fig = plt.figure(figsize=(8, 8),dpi=800)
ax = axisartist.Subplot(fig, 111) 
fig.add_axes(ax)
# 使用 set_visible 方法隐藏绘图区原有所有坐标轴
ax.axis[:].set_visible(False)
# 使用 ax.new_floating_axis 添加新坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
# 给 x 轴添加箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
# 设置刻度显示方向，x 轴为下方显示，y 轴为右侧显示
ax.axis["x"].set_axis_direction("bottom")
ax.axis["y"].set_axis_direction("right")
# x 轴范围为 -10 ～1 0，且分割为 100 份
x=np.linspace(-10,10,100)
y=x**2+3
plt.xlim(-12,12)
plt.ylim(-10, 100)
plt.plot(x,y)
plt.show()


# 梯度下降过程
# 定义一维函数 f(x)=x2+3 的梯度或导数 df/dx=2x
def grad_1(x):
    return 2*x
    
# 定义梯度下降函数 
def grad_descent(grad,x_current,learning_rate,precision,iters_max):        
    for i in range(iters_max):        
        print('第',i,'次迭代x值为:',x_current)        
        grad_current = grad(x_current)        
        if abs(grad_current) < precision:            
            break # 当梯度值小于阈值时，停止迭代         
        else:            
            x_current = x_current - grad_current*learning_rate    
    print('极小值处x为：',x_current)    
    return x_current

#执行模块，赋值运行
if __name__=='__main__': 
    grad_descent(grad_1,x_current=5,learning_rate=0.1,precision=0.000001,iters_max=10000)