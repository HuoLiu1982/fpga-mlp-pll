import numpy as np
import torch
import struct
from train_model import SignalClassifier
def verify_weight_loading():
    """验证权重加载顺序是否正确"""
    
    # 加载训练好的模型
    checkpoint = torch.load('final_signal_classifier_optimized.pth', map_location='cpu')
    model = SignalClassifier(input_size=64, hidden_sizes=[32,16], num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取各层权重
    layer1_weight = model.feature_layers[0].weight.detach().numpy()  # [32, 64]
    layer2_weight = model.feature_layers[4].weight.detach().numpy()  # [16, 32]
    output_weight = model.classifier.weight.detach().numpy()         # [4, 16]
    
    # 读取ROM权重文件
    with open('mlp_weights_q10_22_hex.dat', 'r') as f:
        rom_lines = f.readlines()
    
    # 将ROM权重转换为numpy数组
    rom_weights = []
    for line in rom_lines:
        hex_val = line.strip()
        if hex_val:
            # 将十六进制转换为有符号整数
            int_val = int(hex_val, 16)
            if int_val >= 0x80000000:  # 负数处理
                int_val -= 0x100000000
            rom_weights.append(int_val)
    
    rom_weights = np.array(rom_weights)
    
    print("=== 权重加载验证 ===")
    print(f"ROM权重数量: {len(rom_weights)}")
    print(f"期望数量: 32×64 = {32*64}")
    
    # 验证第一层权重顺序
    Q_FRAC_BITS = 22
    scale = 1 << Q_FRAC_BITS
    
    def float_to_q10_22(x):
        return int(round(x * scale))
    
    # 检查第一个神经元的第一个权重
    expected_first = float_to_q10_22(layer1_weight[0, 0])
    actual_first = rom_weights[0]
    
    print(f"\n第一个权重验证:")
    print(f"  期望: {expected_first:08x} ({expected_first})")
    print(f"  实际: {actual_first:08x} ({actual_first})")
    print(f"  匹配: {'✅' if expected_first == actual_first else '❌'}")
    
    # 检查第一个神经元的最后一个权重
    expected_last_neuron0 = float_to_q10_22(layer1_weight[0, 63])
    actual_last_neuron0 = rom_weights[63]
    
    print(f"\n第一个神经元最后一个权重验证:")
    print(f"  期望: {expected_last_neuron0:08x}")
    print(f"  实际: {actual_last_neuron0:08x}")
    print(f"  匹配: {'✅' if expected_last_neuron0 == actual_last_neuron0 else '❌'}")
    
    # 检查第二个神经元的第一个权重
    expected_first_neuron1 = float_to_q10_22(layer1_weight[1, 0])
    actual_first_neuron1 = rom_weights[64]
    
    print(f"\n第二个神经元第一个权重验证:")
    print(f"  期望: {expected_first_neuron1:08x}")
    print(f"  实际: {actual_first_neuron1:08x}")
    print(f"  匹配: {'✅' if expected_first_neuron1 == actual_first_neuron1 else '❌'}")
    
    # 检查最后一个权重
    expected_final = float_to_q10_22(layer1_weight[31, 63])
    actual_final = rom_weights[2047]
    
    print(f"\n最后一个权重验证:")
    print(f"  期望: {expected_final:08x}")
    print(f"  实际: {actual_final:08x}")
    print(f"  匹配: {'✅' if expected_final == actual_final else '❌'}")
    
    # 批量验证随机样本
    print(f"\n随机抽样验证:")
    mismatches = 0
    for i in range(10):
        neuron = np.random.randint(0, 32)
        input_idx = np.random.randint(0, 64)
        rom_addr = neuron * 64 + input_idx
        
        expected = float_to_q10_22(layer1_weight[neuron, input_idx])
        actual = rom_weights[rom_addr]
        
        match = expected == actual
        if not match:
            mismatches += 1
            
        print(f"  神经元{neuron}, 输入{input_idx}: {'✅' if match else '❌'}")
    
    print(f"\n抽样匹配率: {(10-mismatches)/10*100:.1f}%")
    
    return mismatches == 0

if __name__ == "__main__":
    success = verify_weight_loading()
    print(f"\n权重加载验证: {'通过' if success else '失败'}")