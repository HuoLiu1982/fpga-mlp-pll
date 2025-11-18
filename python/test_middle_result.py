import numpy as np
import torch
import json
import time
import random
from train_model import SignalClassifier

class FPGAEmulatorFixed:
    """修复后的FPGA推理模块Python仿真器（支持中间结果输出）"""
    
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
        
        # 中间结果存储
        self.intermediate_results = {}
        
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
        normalized_signal = np.clip(signal_array / 255.0, 0, 1)    
        return normalized_signal
    
    def fpga_forward_with_debug(self, input_data):
        """FPGA推理前向传播（带详细调试信息）"""
        # 预处理输入数据
        processed_input = self.preprocess_signal(input_data)
        
        # 将输入转换为Q10.22格式
        q_input = np.array([self.float_to_q10_22(x) for x in processed_input], dtype=np.int32)
        
        # 存储中间结果
        self.intermediate_results = {
            'input_normalized': processed_input.copy(),
            'input_q10_22': q_input.copy(),
            'layer1_mac': [],
            'layer1_bias': [],
            'layer1_bn': [],
            'layer1_relu': [],
            'layer2_mac': [],
            'layer2_bias': [],
            'layer2_bn': [],
            'layer2_relu': [],
            'output_mac': [],
            'output_bias': []
        }
        
        print("=" * 80)
        print("FPGA推理过程详细调试信息")
        print("=" * 80)
        
        # 第一层: Linear -> BatchNorm -> ReLU
        print("\n>>> 第一层处理 (64->32)")
        layer1_out = np.zeros(self.HIDDEN1_SIZE, dtype=np.int32)
        
        for i in range(self.HIDDEN1_SIZE):
            print(f"\n--- 神经元 {i} ---")
            
            # MAC操作
            mac_result = 0
            mac_details = []
            for j in range(self.INPUT_SIZE):
                weight = self.q_weights['layer1_weight'][i, j]
                multiply_result = self.fixed_multiply(weight, q_input[j])
                mac_result += multiply_result
                
                if j < 5 or j >= 59:  # 只打印部分结果避免输出过多
                    mac_details.append({
                        'input_idx': j,
                        'input_value': q_input[j],
                        'weight': weight,
                        'multiply_result': multiply_result,
                        'accumulator': mac_result
                    })
            
            self.intermediate_results['layer1_mac'].append({
                'neuron_idx': i,
                'mac_result': mac_result,
                'details': mac_details
            })
            
            print(f"  MAC结果: {mac_result} (浮点: {self.q10_22_to_float(mac_result):.6f})")
            
            # 添加偏置
            bias = self.q_weights['layer1_bias'][i]
            linear_out = mac_result + bias
            self.intermediate_results['layer1_bias'].append({
                'neuron_idx': i,
                'bias': bias,
                'linear_out': linear_out
            })
            print(f"  加偏置: {bias} -> 线性输出: {linear_out} (浮点: {self.q10_22_to_float(linear_out):.6f})")
            
            # 简化的BatchNorm（使用预计算参数）
            scale_q = self.float_to_q10_22(self.bn1_scale[i])
            shift_q = self.float_to_q10_22(self.bn1_shift[i])
            bn_out = self.batchnorm_simple(linear_out, scale_q, shift_q)
            self.intermediate_results['layer1_bn'].append({
                'neuron_idx': i,
                'scale': scale_q,
                'shift': shift_q,
                'bn_out': bn_out
            })
            print(f"  BN: scale={scale_q}, shift={shift_q} -> BN输出: {bn_out} (浮点: {self.q10_22_to_float(bn_out):.6f})")
            
            # ReLU
            relu_out = self.relu_fixed(bn_out)
            layer1_out[i] = relu_out
            self.intermediate_results['layer1_relu'].append({
                'neuron_idx': i,
                'relu_out': relu_out
            })
            print(f"  ReLU输出: {relu_out} (浮点: {self.q10_22_to_float(relu_out):.6f})")
        
        # 第二层: Linear -> BatchNorm -> ReLU
        print("\n>>> 第二层处理 (32->16)")
        layer2_out = np.zeros(self.HIDDEN2_SIZE, dtype=np.int32)
        
        for i in range(self.HIDDEN2_SIZE):
            print(f"\n--- 神经元 {i} ---")
            
            # MAC操作
            mac_result = 0
            mac_details = []
            for j in range(self.HIDDEN1_SIZE):
                weight = self.q_weights['layer2_weight'][i, j]
                multiply_result = self.fixed_multiply(weight, layer1_out[j])
                mac_result += multiply_result
                
                if j < 3 or j >= 29:  # 只打印部分结果
                    mac_details.append({
                        'input_idx': j,
                        'input_value': layer1_out[j],
                        'weight': weight,
                        'multiply_result': multiply_result,
                        'accumulator': mac_result
                    })
            
            self.intermediate_results['layer2_mac'].append({
                'neuron_idx': i,
                'mac_result': mac_result,
                'details': mac_details
            })
            print(f"  MAC结果: {mac_result} (浮点: {self.q10_22_to_float(mac_result):.6f})")
            
            # 添加偏置
            bias = self.q_weights['layer2_bias'][i]
            linear_out = mac_result + bias
            self.intermediate_results['layer2_bias'].append({
                'neuron_idx': i,
                'bias': bias,
                'linear_out': linear_out
            })
            print(f"  加偏置: {bias} -> 线性输出: {linear_out} (浮点: {self.q10_22_to_float(linear_out):.6f})")
            
            # 简化的BatchNorm（使用预计算参数）
            scale_q = self.float_to_q10_22(self.bn2_scale[i])
            shift_q = self.float_to_q10_22(self.bn2_shift[i])
            bn_out = self.batchnorm_simple(linear_out, scale_q, shift_q)
            self.intermediate_results['layer2_bn'].append({
                'neuron_idx': i,
                'scale': scale_q,
                'shift': shift_q,
                'bn_out': bn_out
            })
            print(f"  BN: scale={scale_q}, shift={shift_q} -> BN输出: {bn_out} (浮点: {self.q10_22_to_float(bn_out):.6f})")
            
            # ReLU
            relu_out = self.relu_fixed(bn_out)
            layer2_out[i] = relu_out
            self.intermediate_results['layer2_relu'].append({
                'neuron_idx': i,
                'relu_out': relu_out
            })
            print(f"  ReLU输出: {relu_out} (浮点: {self.q10_22_to_float(relu_out):.6f})")
        
        # 输出层: Linear
        print("\n>>> 输出层处理 (16->4)")
        output_out = np.zeros(self.OUTPUT_SIZE, dtype=np.int32)
        
        for i in range(self.OUTPUT_SIZE):
            print(f"\n--- 输出神经元 {i} ---")
            
            # MAC操作
            mac_result = 0
            mac_details = []
            for j in range(self.HIDDEN2_SIZE):
                weight = self.q_weights['output_weight'][i, j]
                multiply_result = self.fixed_multiply(weight, layer2_out[j])
                mac_result += multiply_result
                
                if j < 3 or j >= 13:  # 只打印部分结果
                    mac_details.append({
                        'input_idx': j,
                        'input_value': layer2_out[j],
                        'weight': weight,
                        'multiply_result': multiply_result,
                        'accumulator': mac_result
                    })
            
            self.intermediate_results['output_mac'].append({
                'neuron_idx': i,
                'mac_result': mac_result,
                'details': mac_details
            })
            print(f"  MAC结果: {mac_result} (浮点: {self.q10_22_to_float(mac_result):.6f})")
            
            # 添加偏置
            bias = self.q_weights['output_bias'][i]
            final_out = mac_result + bias
            output_out[i] = final_out
            self.intermediate_results['output_bias'].append({
                'neuron_idx': i,
                'bias': bias,
                'final_out': final_out
            })
            print(f"  加偏置: {bias} -> 最终输出: {final_out} (浮点: {self.q10_22_to_float(final_out):.6f})")
        
        # 找到最大值（argmax）
        max_idx = np.argmax(output_out)
        confidence = output_out[max_idx]
        
        print("\n" + "=" * 80)
        print("推理结果汇总")
        print("=" * 80)
        print(f"各类别输出值:")
        for i in range(4):
            fpga_val = self.q10_22_to_float(output_out[i])
            print(f"  {self.class_names[i]}: {output_out[i]} (浮点: {fpga_val:.6f})")
        print(f"预测类别: {max_idx} ({self.class_names[max_idx]})")
        print(f"置信度: {confidence} (浮点: {self.q10_22_to_float(confidence):.6f})")
        
        return max_idx, confidence, output_out, self.intermediate_results
    
    def pytorch_forward_with_debug(self, input_data):
        """PyTorch浮点前向传播（带详细调试信息）"""
        print("\n" + "=" * 80)
        print("PyTorch浮点推理过程")
        print("=" * 80)
        
        processed_input = self.preprocess_signal(input_data)
        input_tensor = torch.tensor(processed_input, dtype=torch.float32).unsqueeze(0)
        
        # 手动执行每层以获取中间结果
        x = input_tensor
        
        # 第一层
        print("\n>>> 第一层处理 (64->32)")
        layer1_linear = torch.nn.functional.linear(x, 
            self.state_dict['feature_layers.0.weight'], 
            self.state_dict['feature_layers.0.bias'])
        print(f"线性层输出范围: [{layer1_linear.min():.6f}, {layer1_linear.max():.6f}]")
        
        layer1_bn = torch.nn.functional.batch_norm(layer1_linear, 
            self.state_dict['feature_layers.1.running_mean'],
            self.state_dict['feature_layers.1.running_var'],
            self.state_dict['feature_layers.1.weight'],
            self.state_dict['feature_layers.1.bias'], training=False)
        print(f"BN层输出范围: [{layer1_bn.min():.6f}, {layer1_bn.max():.6f}]")
        
        layer1_relu = torch.relu(layer1_bn)
        print(f"ReLU输出范围: [{layer1_relu.min():.6f}, {layer1_relu.max():.6f}]")
        
        # 第二层
        print("\n>>> 第二层处理 (32->16)")
        layer2_linear = torch.nn.functional.linear(layer1_relu, 
            self.state_dict['feature_layers.4.weight'], 
            self.state_dict['feature_layers.4.bias'])
        print(f"线性层输出范围: [{layer2_linear.min():.6f}, {layer2_linear.max():.6f}]")
        
        layer2_bn = torch.nn.functional.batch_norm(layer2_linear, 
            self.state_dict['feature_layers.5.running_mean'],
            self.state_dict['feature_layers.5.running_var'],
            self.state_dict['feature_layers.5.weight'],
            self.state_dict['feature_layers.5.bias'], training=False)
        print(f"BN层输出范围: [{layer2_bn.min():.6f}, {layer2_bn.max():.6f}]")
        
        layer2_relu = torch.relu(layer2_bn)
        print(f"ReLU输出范围: [{layer2_relu.min():.6f}, {layer2_relu.max():.6f}]")
        
        # 输出层
        print("\n>>> 输出层处理 (16->4)")
        output = torch.nn.functional.linear(layer2_relu, 
            self.state_dict['classifier.weight'], 
            self.state_dict['classifier.bias'])
        print(f"输出层范围: [{output.min():.6f}, {output.max():.6f}]")
        
        max_val, predicted = torch.max(output, 1)
        
        print("\n" + "=" * 80)
        print("PyTorch推理结果汇总")
        print("=" * 80)
        print(f"各类别输出值:")
        for i in range(4):
            print(f"  {self.class_names[i]}: {output[0, i]:.6f}")
        print(f"预测类别: {predicted.item()} ({self.class_names[predicted.item()]})")
        print(f"置信度: {max_val.item():.6f}")
        
        return predicted.item(), max_val.item(), output.numpy()[0]
    
    def compare_results(self, fpga_results, pytorch_results):
        """比较FPGA和PyTorch的结果"""
        print("\n" + "=" * 80)
        print("FPGA vs PyTorch 结果对比")
        print("=" * 80)
        
        fpga_pred, fpga_conf, fpga_outputs, intermediate = fpga_results
        pytorch_pred, pytorch_conf, pytorch_outputs = pytorch_results
        
        print(f"预测结果: FPGA={fpga_pred}({self.class_names[fpga_pred]}), "
              f"PyTorch={pytorch_pred}({self.class_names[pytorch_pred]})")
        
        if fpga_pred == pytorch_pred:
            print("✅ 预测结果一致")
        else:
            print("❌ 预测结果不一致!")
        
        print("\n输出层数值对比:")
        max_error = 0
        for i in range(4):
            fpga_float = self.q10_22_to_float(fpga_outputs[i])
            pytorch_float = pytorch_outputs[i]
            error = abs(fpga_float - pytorch_float)
            if error > max_error:
                max_error = error
            print(f"  {self.class_names[i]}: FPGA={fpga_float:.6f}, PyTorch={pytorch_float:.6f}, "
                  f"误差={error:.6f}")
        
        print(f"\n最大输出层误差: {max_error:.6f}")
        
        return fpga_pred == pytorch_pred, max_error

def classify_user_signal_with_debug():
    """对用户提供的信号数据进行分类（带调试信息）"""
    # 用户提供的64点数据
    signal_str = "178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78"
    
    # 转换为数值列表
    signal_data = [int(x) for x in signal_str.split(',')]
    
    print(f"输入信号数据: {len(signal_data)}点")
    print(f"数据范围: {min(signal_data)} ~ {max(signal_data)}")
    
    # 初始化FPGA仿真器
    fpga_emulator = FPGAEmulatorFixed('final_signal_classifier_optimized.pth')
    
    # 执行FPGA推理（带调试）
    fpga_results = fpga_emulator.fpga_forward_with_debug(signal_data)
    
    # 执行PyTorch推理（带调试）
    pytorch_results = fpga_emulator.pytorch_forward_with_debug(signal_data)
    
    # 对比结果
    is_match, max_error = fpga_emulator.compare_results(fpga_results, pytorch_results)
    
    return fpga_results[0], fpga_emulator.class_names[fpga_results[0]], is_match, max_error

if __name__ == "__main__":
    # 生成LUT和预计算参数
    try:
        from pctofpga import generate_all_luts
        generate_all_luts('final_signal_classifier_optimized.pth')
    except ImportError:
        print("警告: 无法导入pctofpga模块，跳过LUT生成")
    
    print("=" * 80)
    print("FPGA信号分类器 - 详细调试与误差分析")
    print("=" * 80)

    print("\n执行用户信号分类（带详细调试信息）...")
    class_id, class_name, is_match, max_error = classify_user_signal_with_debug()
    
    print(f"\n最终分类结果: {class_name} (类别{class_id})")
    print(f"FPGA与PyTorch一致性: {'匹配' if is_match else '不匹配'}")
    print(f"最大输出层误差: {max_error:.6f}")