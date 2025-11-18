import numpy as np
import torch
import json
import time
import random
from train_model import SignalClassifier

class FPGAEmulatorFixed:
    """修复后的FPGA推理模块Python仿真器（支持多组数据测试）"""
    
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
        
        # 类别标签映射
        self.class_names = {
            0: "正弦波",
            1: "方波", 
            2: "三角波",
            3: "其他信号"
        }
        
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
    def preprocess_signal(self, signal_data):
        """预处理输入信号数据"""
        # 转换为numpy数组
        signal_array = np.array(signal_data, dtype=np.float32)
        # 归一化到0-1之间
        normalized_signal = np.clip(signal_array/ 255.0, 0, 1)    
        return normalized_signal
    
    def fpga_forward(self, input_data):
        """FPGA推理前向传播"""
        # 预处理输入数据
        processed_input = self.preprocess_signal(input_data)
        
        # 将输入转换为Q10.22格式
        q_input = np.array([self.float_to_q10_22(x) for x in processed_input], dtype=np.int32)
        
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
        processed_input = self.preprocess_signal(input_data)
        input_tensor = torch.tensor(processed_input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            # 使用logits而不是softmax，与FPGA保持一致
            max_val, predicted = torch.max(output, 1)
        
        return predicted.item(), max_val.item(), output.numpy()[0]
    
    def classify_signal(self, signal_data, verbose=True):
        """对输入信号进行分类"""
        if verbose:
            print("=" * 60)
            print("信号分类结果")
            print("=" * 60)
        
        # 检查输入数据长度
        if len(signal_data) != 64:
            print(f"警告: 输入数据长度应为64，当前为{len(signal_data)}")
            if len(signal_data) > 64:
                signal_data = signal_data[:64]
                if verbose:
                    print(f"已截取前64个点")
            else:
                print(f"数据长度不足，无法处理")
                return None, None
        
        # FPGA定点推理
        fpga_pred, fpga_conf, fpga_outputs = self.fpga_forward(signal_data)
        
        # PyTorch浮点推理（作为参考）
        pytorch_pred, pytorch_conf, pytorch_outputs = self.pytorch_forward(signal_data)
        
        if verbose:
            # 输出结果
            print(f"输入信号: 64点数据")
            print(f"FPGA定点推理结果: {self.class_names[fpga_pred]} (类别{fpga_pred})")
            print(f"PyTorch浮点推理结果: {self.class_names[pytorch_pred]} (类别{pytorch_pred})")
            print(f"置信度: FPGA={self.q10_22_to_float(fpga_conf):.4f}, PyTorch={pytorch_conf:.4f}")
            
            print("\n各类别输出值:")
            for i in range(4):
                fpga_val = self.q10_22_to_float(fpga_outputs[i])
                pytorch_val = pytorch_outputs[i]
                print(f"  {self.class_names[i]}: FPGA={fpga_val:8.4f}, PyTorch={pytorch_val:8.4f}")
            
            # 判断是否一致
            if fpga_pred == pytorch_pred:
                print(f"\n✅ FPGA与PyTorch结果一致: {self.class_names[fpga_pred]}")
            else:
                print(f"\n⚠️  FPGA与PyTorch结果不一致")
        
        return fpga_pred, self.class_names[fpga_pred]

def classify_user_signal():
    """对用户提供的信号数据进行分类"""
    # 用户提供的64点数据
    signal_str = "160, 157, 153, 150, 146, 143, 139, 135, 132, 128, 125, 121, 117, 114, 110, 107, 103, 99, 96, 92, 89, 85, 81, 78, 74, 71, 67, 63, 60, 56, 53, 57, 66, 74, 82, 91, 99, 108, 116, 124, 132, 141, 149, 158, 166, 175, 183, 191, 200, 200, 196, 192, 189, 185, 182, 178, 175, 171, 168, 164, 161, 157, 153, 150"
    
    # 转换为数值列表
    signal_data = [int(x) for x in signal_str.split(',')]
    
    print(f"输入信号数据: {len(signal_data)}点")
    print(f"数据范围: {min(signal_data)} ~ {max(signal_data)}")
    
    # 初始化FPGA仿真器
    fpga_emulator = FPGAEmulatorFixed('final_signal_classifier_adaptive.pth')
    
    # 进行分类
    result_class, class_name = fpga_emulator.classify_signal(signal_data)
    
    return result_class, class_name

if __name__ == "__main__":
    # 生成LUT和预计算参数
    try:
        from pctofpga import generate_all_luts
        generate_all_luts('final_signal_classifier_adaptive.pth')
    except ImportError:
        print("警告: 无法导入pctofpga模块，跳过LUT生成")
    
    print("=" * 70)
    print("FPGA信号分类器 - 多组数据测试与ADC模拟")
    print("=" * 70)

    print("\n执行单次用户信号分类...")
    class_id, class_name = classify_user_signal()
    print(f"\n最终分类结果: {class_name} (类别{class_id})")
            
 