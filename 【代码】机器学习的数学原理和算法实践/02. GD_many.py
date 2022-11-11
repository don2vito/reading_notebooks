import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# 作函数图像
x,y = np.mgrid[-2:2:20j,-2:2:20j]
# 测试数据
z = (x**2 + y**2)
# 三维图形
fig = plt.figure(figsize=(8, 8),dpi=800)
ax = plt.subplot(111, projection='3d')
fig.add_axes(ax)
ax.set_title('f(x,y)=x^2+y^2')
ax.plot_surface(x,y,z,rstride=9,cstride=1, cmap=plt.cm.Blues_r)
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# 梯度下降过程
# 定义多元函数 f(x,y)=x2+y2 的梯度 f'(x)=2x, f'(y)=2y
def grad_2(p):    
    derivx = 2*p[0]    
    derivy = 2*p[1]    
    return np.array([derivx, derivy])
    
# 定义梯度下降函数 
def grad_descent(grad,p_current,learning_rate,precision,iters_max):            
    for i in range(iters_max):        
        print ('第',i,'次迭代p值为:',p_current)        
        grad_current = grad(p_current)        
        if np.linalg.norm(grad_current, ord=2) < precision:            
            break  # 当梯度趋近于 0 时，视为收敛         
        else:            
            p_current = p_current - grad_current*learning_rate              
    print ('极小值处p为：',p_current)    
    return p_current
    
#执行模块
if __name__=='__main__':    
    grad_descent(grad_2,p_current=np.array([1,-1]),learning_rate=0.1,precision=0.000001,iters_max=10000) 