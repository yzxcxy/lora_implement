import torch
import torch.nn as nn

class LoraLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoraLayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())    
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        # LoRA 部分的前向传播
        return self.alpha * (x @ self.A @ self.B)


class LayerWithLora(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super(LayerWithLora, self).__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)
        self.merged = False  # 标识是否已经合并 LoRA 参数
        self.original_weight = None  # 存储原始的线性层权重
    
    def forward(self, x):
        if self.merged:
            # 如果已经合并了 LoRA 参数，直接使用合并后的线性层
            return self.linear(x)
        else:
            # 训练阶段：原始线性层 + LoRA 层的输出
            return self.linear(x) + self.lora(x)

    def merge_lora(self):
        # 将 LoRA 参数与原始线性层的权重合并
        if not self.merged:
            self.original_weight = self.linear.weight.data.clone()  # 存储原始权重
            self.linear.weight.data += (self.lora.A @ self.lora.B).t() * self.lora.alpha
            self.merged = True
    
    def unmerge_lora(self):
        # 取消合并 LoRA 参数，恢复到原始的线性层权重
        if self.merged and self.original_weight is not None:
            self.linear.weight.data = self.original_weight  # 恢复原始权重
            self.merged = False


# 动态给所有的线性层注入 LoraLayer
def apply_lora(net, rank, alpha):
    for name, module in net.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(net, name, LayerWithLora(module, rank, alpha))
        else:
            apply_lora(module, rank, alpha)
    return net

# 在训练阶段，冻结原模型的参数
def freeze(model):
    for name, param in model.named_parameters():
        # 检查参数是否属于 lora 子层
        if 'lora' in name:
            param.requires_grad = True  # LoRA 参数不被冻结
        else:
            param.requires_grad = False  # 冻结其他所有参数

# 推理阶段合并 LoRA 参数
def merge_all_lora_layers(model):
    for module in model.modules():
        if isinstance(module, LayerWithLora):
            module.merge_lora()

# 取消所有层的 LoRA 参数合并，恢复原始状态
def unmerge_all_lora_layers(model):
    for module in model.modules():
        if isinstance(module, LayerWithLora):
            module.unmerge_lora()


# 在推理时只使用合并后的单个矩阵
# 注意这里不应该为该文件的导出函数，因为它依赖于上面的函数，这只是一个测试函数
def inference(model, x):
    model.eval()  # 进入推理模式
    merge_all_lora_layers(model)  # 合并LoRA和原始权重
    with torch.no_grad():  # 确保推理阶段没有梯度计算
        output = model(x)
    return output


# 测试打印模型结构# 假设我们已经定义好了模型并应用了 LoRA 层
# model = nn.Sequential(
#     nn.Linear(10, 5),  # 线性层
#     nn.ReLU(),
#     nn.Linear(5, 2)    # 线性层
# )

# # 应用 LoRA 层
# apply_lora(model, rank=2, alpha=1.0)

# 打印模型结构
# print(model)
# 打印详细的模型结构，包括每层的名称和参数
# for name, module in model.named_modules():
#     print(f"Layer: {name}")
#     print(module)
#     print("--------------")

# # 打印模型中的所有参数
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
#     print("--------------")


