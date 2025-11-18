import numpy as np
import torch
import json
from train_model import SignalClassifier

def precompute_batchnorm_params(model_path, output_file='batchnorm_params.json'):
    """预计算BatchNorm的scale和shift参数"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SignalClassifier(
        input_size=64,
        hidden_sizes=[32, 16],
        num_classes=4,
        dropout_rate=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    state_dict = model.state_dict()
    
    # Q10.22格式参数
    Q_INT_BITS = 10
    Q_FRAC_BITS = 22
    Q_TOTAL_BITS = 32
    
    def float_to_q10_22(value):
        scale = 1 << Q_FRAC_BITS
        q_value = int(round(value * scale))
        max_val = (1 << (Q_TOTAL_BITS - 1)) - 1
        min_val = -(1 << (Q_TOTAL_BITS - 1))
        return max(min(q_value, max_val), min_val)
    
    # 预计算BatchNorm参数
    bn_params = {}
    
    # 第一层BatchNorm参数
    bn1_weight = state_dict['feature_layers.1.weight'].detach().numpy()
    bn1_bias = state_dict['feature_layers.1.bias'].detach().numpy()
    bn1_running_mean = state_dict['feature_layers.1.running_mean'].detach().numpy()
    bn1_running_var = state_dict['feature_layers.1.running_var'].detach().numpy()
    
    epsilon = 1e-5
    bn1_scale = bn1_weight / np.sqrt(bn1_running_var + epsilon)
    bn1_shift = bn1_bias - bn1_scale * bn1_running_mean
    
    bn_params['bn1_scale'] = [float_to_q10_22(x) for x in bn1_scale]
    bn_params['bn1_shift'] = [float_to_q10_22(x) for x in bn1_shift]
    
    # 第二层BatchNorm参数
    bn2_weight = state_dict['feature_layers.5.weight'].detach().numpy()
    bn2_bias = state_dict['feature_layers.5.bias'].detach().numpy()
    bn2_running_mean = state_dict['feature_layers.5.running_mean'].detach().numpy()
    bn2_running_var = state_dict['feature_layers.5.running_var'].detach().numpy()
    
    bn2_scale = bn2_weight / np.sqrt(bn2_running_var + epsilon)
    bn2_shift = bn2_bias - bn2_scale * bn2_running_mean
    
    bn_params['bn2_scale'] = [float_to_q10_22(x) for x in bn2_scale]
    bn_params['bn2_shift'] = [float_to_q10_22(x) for x in bn2_shift]
    
    # 保存预计算参数
    with open(output_file, 'w') as f:
        json.dump(bn_params, f, indent=2)
    
    print(f"BatchNorm参数预计算完成，保存到: {output_file}")
    
    # 打印数值范围
    print("\nBatchNorm参数数值范围:")
    for key, values in bn_params.items():
        float_values = [x / (1 << Q_FRAC_BITS) for x in values]
        print(f"  {key}: min={min(float_values):.4f}, max={max(float_values):.4f}, mean={np.mean(float_values):.4f}")
    
    return bn_params

def generate_all_luts(model_path, output_dir='.'):
    """生成所有LUT的Verilog代码"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SignalClassifier(
        input_size=64,
        hidden_sizes=[32, 16],
        num_classes=4,
        dropout_rate=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    state_dict = model.state_dict()
    
    # Q10.22格式参数
    Q_INT_BITS = 10
    Q_FRAC_BITS = 22
    Q_TOTAL_BITS = 32
    
    def float_to_q10_22(value):
        scale = 1 << Q_FRAC_BITS
        q_value = int(round(value * scale))
        max_val = (1 << (Q_TOTAL_BITS - 1)) - 1
        min_val = -(1 << (Q_TOTAL_BITS - 1))
        return max(min(q_value, max_val), min_val)
    
    # 预计算BatchNorm参数
    bn_params = precompute_batchnorm_params(model_path)
    
    # 生成LUT Verilog文件
    with open(f'{output_dir}/mlp_luts.v', 'w') as f:
        f.write('`timescale 1ns / 1ps\n\n')
        f.write('// MLP LUTs for FPGA Implementation\n')
        f.write('// Auto-generated from PyTorch model\n')
        f.write('// Q10.22 fixed-point format\n\n')
        
        # Layer1偏置LUT
        f.write('module layer1_bias_lut(\n')
        f.write('    input [4:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        bias1 = state_dict['feature_layers.0.bias'].detach().numpy()
        for i in range(32):
            fixed_value = float_to_q10_22(bias1[i])
            f.write(f'        5\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        # Layer2权重LUT
        f.write('module layer2_weight_lut(\n')
        f.write('    input [9:0] addr,  // 16*32=512个权重\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        weight2 = state_dict['feature_layers.4.weight'].detach().numpy().flatten()
        for i in range(512):
            fixed_value = float_to_q10_22(weight2[i])
            f.write(f'        10\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        # Layer2偏置LUT
        f.write('module layer2_bias_lut(\n')
        f.write('    input [3:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        bias2 = state_dict['feature_layers.4.bias'].detach().numpy()
        for i in range(16):
            fixed_value = float_to_q10_22(bias2[i])
            f.write(f'        4\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        # 输出层权重LUT
        f.write('module output_weight_lut(\n')
        f.write('    input [5:0] addr,  // 4*16=64个权重\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        weight_out = state_dict['classifier.weight'].detach().numpy().flatten()
        for i in range(64):
            fixed_value = float_to_q10_22(weight_out[i])
            f.write(f'        6\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        # 输出层偏置LUT
        f.write('module output_bias_lut(\n')
        f.write('    input [1:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        bias_out = state_dict['classifier.bias'].detach().numpy()
        for i in range(4):
            fixed_value = float_to_q10_22(bias_out[i])
            f.write(f'        2\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        # 预计算的BatchNorm参数LUT
        f.write('module bn1_scale_lut(\n')
        f.write('    input [4:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        for i in range(32):
            fixed_value = bn_params['bn1_scale'][i]
            f.write(f'        5\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        f.write('module bn1_shift_lut(\n')
        f.write('    input [4:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        for i in range(32):
            fixed_value = bn_params['bn1_shift'][i]
            f.write(f'        5\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        f.write('module bn2_scale_lut(\n')
        f.write('    input [3:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        for i in range(16):
            fixed_value = bn_params['bn2_scale'][i]
            f.write(f'        4\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n\n')
        
        f.write('module bn2_shift_lut(\n')
        f.write('    input [3:0] addr,\n')
        f.write('    output reg [31:0] data\n')
        f.write(');\n\n')
        f.write('always @(*) begin\n')
        f.write('    case(addr)\n')
        
        for i in range(16):
            fixed_value = bn_params['bn2_shift'][i]
            f.write(f'        4\'d{i}: data = 32\'h{fixed_value & 0xFFFFFFFF:08x};\n')
        
        f.write('        default: data = 32\'h00000000;\n')
        f.write('    endcase\n')
        f.write('end\n\n')
        f.write('endmodule\n')
    
    print(f"LUT Verilog文件已生成: {output_dir}/mlp_luts.v")
    
    # 生成权重ROM数据文件
    weight1 = state_dict['feature_layers.0.weight'].detach().numpy().flatten()
    
    import struct
    with open(f'{output_dir}/mlp_weights_q10_22.dat', 'wb') as f:
        #header = b'MLP_WEIGHTS_Q10_22'
        #f.write(header)
        
        for value in weight1:
            fixed_value = float_to_q10_22(value)
            f.write(struct.pack('>i', fixed_value))
        
        print(f"权重ROM文件已生成: {output_dir}/mlp_weights_q10_22.dat")
        print(f"写入 {len(weight1)} 个32位定点数")

if __name__ == "__main__":
    generate_all_luts('final_signal_classifier_optimized.pth')