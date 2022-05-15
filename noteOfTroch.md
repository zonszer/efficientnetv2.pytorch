#####related operation张量的数学运算： 
    torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
out ${i}=\operatorname{input}{i}+$ value $\times \frac{\text { tensor } 1{i}}{\text { tensor } 2{i}}$

    torch.add(tensor1, tensor2, *, alpha=1, out=None)

    torch.addcmul()
计算公式为：out ${i}=$ input ${i}+$ value $\times$ tensor $1{i} \times$ tensor $2{i}$


#####1.4 静态图与动态图机制
计算图是用来描述运算的有向无环图，有两个主要元素：节点 (Node) 和边 (Edge)。节点表示数据，如向量、矩阵、张量。边表示运算，如加减乘除卷积等。
![avatar](https://pic2.zhimg.com/80/v2-464ea7ee4475f3c7f08c389f65fd3e89_1440w.jpg)

#####loss.backward（）中的grad_tensors 参数（多梯度权重计算）:
    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors
    # 最终的 w 的导数由两部分组成。∂y0/∂w * 1 + ∂y1/∂w * 2
    print(w.grad)
    #结果为：tensor([9.])

##careful!b = torch.add(w, b) ， b = b + w与 b.add(w) 最终.grad的结果不同

![avatar](https://pic1.zhimg.com/80/v2-11280c55c7d6f98b4ddb6fffe9c3645c_1440w.jpg)

#####DataLoader 与 DataSet
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)