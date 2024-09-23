import unittest
import torch
import torch.nn as nn
from LoraLayer import LayerWithLora, apply_lora, freeze, merge_all_lora_layers, unmerge_all_lora_layers, inference

class TestLoraLayer(unittest.TestCase):
    def setUp(self):
        # 设置一个简单的模型用于测试
        self.model = nn.Sequential(
            nn.Linear(10, 5),  # 线性层
            nn.ReLU(),
            nn.Linear(5, 2)    # 线性层
        )
        self.rank = 2
        self.alpha = 1.0
        apply_lora(self.model, self.rank, self.alpha)  # 将 LoRA 应用于模型
    
    def test_freeze(self):
        # 验证冻结后的参数，LoRA的参数应该没有被冻结，其他参数应该被冻结
        freeze(self.model)
        
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                self.assertTrue(param.requires_grad)  # LoRA 参数应该是可训练的
            else:
                self.assertFalse(param.requires_grad)  # 原始权重应该被冻结
    
    def test_training_forward(self):
        # 验证训练阶段前向传播：原始权重 + LoRA 的贡献
        x = torch.randn(3, 10)  # 输入数据
        self.model.train()
        output = self.model(x)
        self.assertEqual(output.shape, (3, 2))  # 输出维度应该符合预期
    
    def test_inference(self):
        # 验证推理阶段，权重已经合并且只使用一个矩阵
        x = torch.randn(3, 10)  # 输入数据
        output_train = self.model(x)  # 获取训练阶段的输出
        
        output_infer = inference(self.model, x)  # 进入推理阶段
        self.assertEqual(output_infer.shape, (3, 2))  # 输出维度应该符合预期
        self.assertTrue(torch.equal(output_train, output_infer))  # 确保训练和推理的输出相同

        # 验证是否只用了一个矩阵
        for module in self.model.modules():
            if isinstance(module, LayerWithLora):
                self.assertTrue(module.merged)  # 推理阶段应合并参数

    def test_unmerge(self):
        # 验证取消合并后的行为
        x = torch.randn(3, 10)  # 输入数据
        merge_all_lora_layers(self.model)  # 先合并
        unmerge_all_lora_layers(self.model)  # 然后取消合并
        
        output = self.model(x)  # 验证取消合并后模型仍然工作
        self.assertEqual(output.shape, (3, 2))
        
        # 验证是否正确恢复到未合并状态
        for module in self.model.modules():
            if isinstance(module, LayerWithLora):
                self.assertFalse(module.merged)  # 确保参数未合并


if __name__ == '__main__':
    unittest.main()
