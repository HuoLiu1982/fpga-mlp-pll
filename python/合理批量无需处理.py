import numpy as np
import torch
import json
from train_model import SignalClassifier

class FPGAEmulatorFixed:
    """修复后的FPGA推理模块Python仿真器（Q10.22 + 预计算BatchNorm）"""
    
    def __init__(self, model_path):
        # 加载训练好的模型
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = SignalClassifier(
            input_size=64,
            hidden_sizes=[32, 16],
            num_classes=4,
            dropout_rate=0.2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 提取模型参数
        self.state_dict = self.model.state_dict()
        
        # Q10.22格式参数
        self.Q_INT_BITS = 10
        self.Q_FRAC_BITS = 22
        self.Q_TOTAL_BITS = 32
        
        print(f"使用Q{self.Q_INT_BITS}.{self.Q_FRAC_BITS}格式")
        
        # 网络参数
        self.INPUT_SIZE = 64
        self.HIDDEN1_SIZE = 32
        self.HIDDEN2_SIZE = 16
        self.OUTPUT_SIZE = 4
        
        # 预计算BatchNorm参数
        self.precompute_batchnorm()
        
        # 转换所有权重为Q10.22格式
        self.convert_weights_to_fixed_point()
    
    def float_to_q10_22(self, value):
        """将浮点数转换为Q10.22格式"""
        scale = 1 << self.Q_FRAC_BITS
        q_value = int(round(value * scale))
        # 饱和处理
        max_val = (1 << (self.Q_TOTAL_BITS - 1)) - 1
        min_val = -(1 << (self.Q_TOTAL_BITS - 1))
        return max(min(q_value, max_val), min_val)
    
    def q10_22_to_float(self, q_value):
        """将Q10.22格式转换为浮点数"""
        return q_value / (1 << self.Q_FRAC_BITS)
    
    def precompute_batchnorm(self):
        """预计算BatchNorm参数"""
        print("预计算BatchNorm参数...")
        
        # 第一层BatchNorm参数
        bn1_weight = self.state_dict['feature_layers.1.weight'].detach().numpy()
        bn1_bias = self.state_dict['feature_layers.1.bias'].detach().numpy()
        bn1_running_mean = self.state_dict['feature_layers.1.running_mean'].detach().numpy()
        bn1_running_var = self.state_dict['feature_layers.1.running_var'].detach().numpy()
        
        epsilon = 1e-5
        self.bn1_scale = bn1_weight / np.sqrt(bn1_running_var + epsilon)
        self.bn1_shift = bn1_bias - self.bn1_scale * bn1_running_mean
        
        # 第二层BatchNorm参数
        bn2_weight = self.state_dict['feature_layers.5.weight'].detach().numpy()
        bn2_bias = self.state_dict['feature_layers.5.bias'].detach().numpy()
        bn2_running_mean = self.state_dict['feature_layers.5.running_mean'].detach().numpy()
        bn2_running_var = self.state_dict['feature_layers.5.running_var'].detach().numpy()
        
        self.bn2_scale = bn2_weight / np.sqrt(bn2_running_var + epsilon)
        self.bn2_shift = bn2_bias - self.bn2_scale * bn2_running_mean
        
        print("BatchNorm参数预计算完成")
    
    def convert_weights_to_fixed_point(self):
        """将所有模型权重转换为Q10.22格式"""
        self.q_weights = {}
        
        # 转换各层参数
        layers = [
            ('layer1_weight', 'feature_layers.0.weight'),
            ('layer1_bias', 'feature_layers.0.bias'),
            ('layer2_weight', 'feature_layers.4.weight'),
            ('layer2_bias', 'feature_layers.4.bias'),
            ('output_weight', 'classifier.weight'),
            ('output_bias', 'classifier.bias')
        ]
        
        for q_name, torch_name in layers:
            tensor = self.state_dict[torch_name].detach().numpy()
            self.q_weights[q_name] = np.vectorize(self.float_to_q10_22)(tensor)
            print(f"转换 {q_name:20}: 形状{tensor.shape}")
    
    def fixed_multiply(self, a, b):
        """Q10.22定点数乘法"""
        # Q10.22 * Q10.22 = Q20.44 (64位结果)
        product = np.int64(a) * np.int64(b)
        # 右移22位转换为Q10.22
        return np.int32(product >> self.Q_FRAC_BITS)
    
    def relu_fixed(self, x):
        """ReLU激活函数"""
        return x if x >= 0 else 0
    
    def batchnorm_simple(self, x, scale, shift):
        """简化的BatchNorm计算，使用预计算参数"""
        # y = scale * x + shift
        return self.fixed_multiply(scale, x) + shift
    
    def fpga_forward(self, input_data):
        """FPGA推理前向传播"""
        # 将输入转换为Q10.22格式
        q_input = np.array([self.float_to_q10_22(x) for x in input_data], dtype=np.int32)
        
        # 第一层: Linear -> BatchNorm -> ReLU
        layer1_out = np.zeros(self.HIDDEN1_SIZE, dtype=np.int32)
        for i in range(self.HIDDEN1_SIZE):
            # MAC操作
            mac_result = 0
            for j in range(self.INPUT_SIZE):
                weight = self.q_weights['layer1_weight'][i, j]
                mac_result += self.fixed_multiply(weight, q_input[j])
            
            # 添加偏置
            linear_out = mac_result + self.q_weights['layer1_bias'][i]
            
            # 简化的BatchNorm（使用预计算参数）
            scale_q = self.float_to_q10_22(self.bn1_scale[i])
            shift_q = self.float_to_q10_22(self.bn1_shift[i])
            bn_out = self.batchnorm_simple(linear_out, scale_q, shift_q)
            
            # ReLU
            layer1_out[i] = self.relu_fixed(bn_out)
        
        # 第二层: Linear -> BatchNorm -> ReLU
        layer2_out = np.zeros(self.HIDDEN2_SIZE, dtype=np.int32)
        for i in range(self.HIDDEN2_SIZE):
            # MAC操作
            mac_result = 0
            for j in range(self.HIDDEN1_SIZE):
                weight = self.q_weights['layer2_weight'][i, j]
                mac_result += self.fixed_multiply(weight, layer1_out[j])
            
            # 添加偏置
            linear_out = mac_result + self.q_weights['layer2_bias'][i]
            
            # 简化的BatchNorm（使用预计算参数）
            scale_q = self.float_to_q10_22(self.bn2_scale[i])
            shift_q = self.float_to_q10_22(self.bn2_shift[i])
            bn_out = self.batchnorm_simple(linear_out, scale_q, shift_q)
            
            # ReLU
            layer2_out[i] = self.relu_fixed(bn_out)
        
        # 输出层: Linear
        output_out = np.zeros(self.OUTPUT_SIZE, dtype=np.int32)
        for i in range(self.OUTPUT_SIZE):
            # MAC操作
            mac_result = 0
            for j in range(self.HIDDEN2_SIZE):
                weight = self.q_weights['output_weight'][i, j]
                mac_result += self.fixed_multiply(weight, layer2_out[j])
            
            # 添加偏置
            output_out[i] = mac_result + self.q_weights['output_bias'][i]
        
        # 找到最大值（argmax）
        max_idx = np.argmax(output_out)
        confidence = output_out[max_idx]
        
        return max_idx, confidence, output_out
    
    def pytorch_forward(self, input_data):
        """PyTorch浮点前向传播（参考）"""
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            # 使用logits而不是softmax，与FPGA保持一致
            max_val, predicted = torch.max(output, 1)
        
        return predicted.item(), max_val.item(), output.numpy()[0]

def verify_fixed_implementation():
    """验证修复后的实现"""
    print("开始验证修复后的FPGA实现...")
    
    # 初始化修复后的FPGA仿真器
    fpga_emulator = FPGAEmulatorFixed('final_signal_classifier_optimized.pth')
    
    # 生成测试数据
    from train_model import SignalDataset
    test_dataset = SignalDataset(num_samples=5000, include_other=True)
    test_signals = []
    true_labels = []
    
    for i in range(5000):
        signal, label = test_dataset[i]
        test_signals.append(signal.numpy())
        true_labels.append(label.item())
    
    total_samples = len(test_signals)
    
    # 统计结果
    fpga_correct = 0
    pytorch_correct = 0
    both_correct = 0
    
    print(f"测试样本数量: {total_samples}")
    print("\n开始逐样本验证...")
    
    for i, (signal, true_label) in enumerate(zip(test_signals, true_labels)):
        if i % 1000 == 0:
            print(f"处理样本 {i}/{total_samples}")
        
        # FPGA定点推理
        fpga_pred, fpga_conf, fpga_outputs = fpga_emulator.fpga_forward(signal)
        
        # PyTorch浮点推理
        pytorch_pred, pytorch_conf, pytorch_outputs = fpga_emulator.pytorch_forward(signal)
        
        # 统计正确率
        fpga_correct += 1 if fpga_pred == true_label else 0
        pytorch_correct += 1 if pytorch_pred == true_label else 0
        both_correct += 1 if (fpga_pred == true_label and pytorch_pred == true_label) else 0
        
        # 详细输出前3个样本
        if i < 3:
            print(f"\n样本 {i+1}:")
            print(f"  真实类别: {true_label}")
            print(f"  FPGA预测: {fpga_pred}, 输出值: {fpga_emulator.q10_22_to_float(fpga_conf):.4f}")
            print(f"  PyTorch预测: {pytorch_pred}, 输出值: {pytorch_conf:.4f}")
            
            # 输出各类别输出值对比
            print("  各类别输出值对比:")
            for j in range(4):
                fpga_val = fpga_emulator.q10_22_to_float(fpga_outputs[j])
                pytorch_val = pytorch_outputs[j]
                error = abs(fpga_val - pytorch_val)
                print(f"    类别{j}: FPGA={fpga_val:8.4f}, PyTorch={pytorch_val:8.4f}, 误差={error:8.4f}")
    
    # 统计结果
    fpga_accuracy = fpga_correct / total_samples * 100
    pytorch_accuracy = pytorch_correct / total_samples * 100
    agreement_rate = both_correct / total_samples * 100
    
    print("\n" + "="*60)
    print("修复后的验证结果总结:")
    print("="*60)
    print(f"FPGA定点推理准确率: {fpga_accuracy:.2f}%")
    print(f"PyTorch浮点推理准确率: {pytorch_accuracy:.2f}%")
    print(f"两者一致率: {agreement_rate:.2f}%")
    
    # 精度损失分析
    accuracy_drop = pytorch_accuracy - fpga_accuracy
    print(f"精度下降: {accuracy_drop:.2f}%")
    
    if accuracy_drop < 5.0:
        print("\n✅ FPGA实现验证通过！")
    else:
        print("\n❌ FPGA实现需要进一步优化")
    
    return {
        'fpga_accuracy': fpga_accuracy,
        'pytorch_accuracy': pytorch_accuracy,
        'agreement_rate': agreement_rate,
        'accuracy_drop': accuracy_drop
    }

if __name__ == "__main__":
    # 生成LUT和预计算参数
    from pctofpga import generate_all_luts
    generate_all_luts('final_signal_classifier_optimized.pth')
    
    # 运行修复后的验证
    results = verify_fixed_implementation()
