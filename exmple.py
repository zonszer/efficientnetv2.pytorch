import torch

######--------E1
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y=(x+w)*(w+1)
a = torch.add(w, x)     # retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)
# y 求导
x.backward()        
y.backward()
# y.backward()           #【2】但是在第二次执行y.backward()时会出错。因为 PyTorch 默认是每次求取梯度之后不保存计算图的
# 打印 w 的梯度，就是 y 对 w 的导数
print(w.grad)
print(a.grad)
print(b.grad)
print(x.grad)       #【1】 注意每次更新参数之后，都要清零张量的梯度 x.grad.zero_()

print(w.numpy())
print(w.data.numpy())

######--------E1.2多次求导（最终结果相同）
# 【2->3】第一次求导，设置 retain_graph=True，保留计算图
y.backward(retain_graph=True)
print(w.grad)
# 【3】第二次求导成功
y.backward()
print(w.grad)


######---------E2（二阶导）
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)     # y = x**2
# 【1】 如果需要求 2 阶导，需要设置 create_graph=True，让一阶导数 grad_1 也拥有计算图
grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
print(grad_1)
# 这里求 2 阶导
grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
print(grad_2)

######---------E3 inplace operation
'''以加法来说，inplace 操作有a += x，a.add_(x)，改变后的值和原来的值内存地址是同一个。
非 inplace 操作有a = a + x，a.add(x)，改变后的值和原来的值内存地址不是同一个。'''

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y = (x + w) * (w + 1)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
# 在反向传播之前 inplace 改变了 w 的值，再执行 backward() 会报错
w.add_(1)
y.backward()
