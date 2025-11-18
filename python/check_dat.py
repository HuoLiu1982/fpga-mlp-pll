import struct
import os
import numpy as np

def analyze_weight_file(file_path):
    """分析权重数据文件"""
    print(f"分析文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path)} 字节")
    
    # 读取文件内容
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"文件总字节数: {len(data)}")
    
    # 检查文件格式
    if len(data) % 4 != 0:
        print(f"❌ 错误: 文件大小不是4字节的整数倍")
        return False
    
    num_weights = len(data) // 4
    print(f"权重数量: {num_weights}")
    
    # 解析权重数据
    weights = []
    for i in range(min(10, num_weights)):  # 只显示前10个
        weight_bytes = data[i*4:(i+1)*4]
        # 尝试大端序解析
        weight_big = struct.unpack('>i', weight_bytes)[0]
        # 尝试小端序解析  
        weight_little = struct.unpack('<i', weight_bytes)[0]
        
        weights.append(weight_big)
        
        print(f"权重[{i}]:")
        print(f"  十六进制: {weight_bytes.hex()}")
        print(f"  大端序值: {weight_big} (0x{weight_big & 0xFFFFFFFF:08x})")
        print(f"  小端序值: {weight_little} (0x{weight_little & 0xFFFFFFFF:08x})")
        print(f"  浮点值(Q10.22): {weight_big / (1 << 22):.6f}")
    
    # 检查数据范围
    all_weights = [struct.unpack('>i', data[i*4:(i+1)*4])[0] for i in range(num_weights)]
    min_val = min(all_weights)
    max_val = max(all_weights)
    print(f"\n数据范围:")
    print(f"  最小值: {min_val} (0x{min_val & 0xFFFFFFFF:08x})")
    print(f"  最大值: {max_val} (0x{max_val & 0xFFFFFFFF:08x})")
    
    return True

def convert_to_hex_format(input_file, output_file):
    """转换为十六进制文本格式"""
    print(f"\n转换文件格式...")
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    num_weights = len(data) // 4
    
    with open(output_file, 'w') as f:
        for i in range(num_weights):
            weight_bytes = data[i*4:(i+1)*4]
            weight_val = struct.unpack('>i', weight_bytes)[0]
            # 写入十六进制格式，每行一个
            f.write(f"{weight_val & 0xFFFFFFFF:08x}\n")
    
    print(f"已生成十六进制格式文件: {output_file}")
    print(f"包含 {num_weights} 行数据")

def convert_to_bin_format(input_file, output_file):
    """转换为二进制文本格式"""
    print(f"\n转换文件格式...")
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    num_weights = len(data) // 4
    
    with open(output_file, 'w') as f:
        for i in range(num_weights):
            weight_bytes = data[i*4:(i+1)*4]
            weight_val = struct.unpack('>i', weight_bytes)[0]
            # 写入二进制格式，每行一个32位二进制数
            f.write(f"{weight_val & 0xFFFFFFFF:032b}\n")
    
    print(f"已生成二进制格式文件: {output_file}")
    print(f"包含 {num_weights} 行数据")

def check_rom_configuration():
    """检查ROM配置建议"""
    print(f"\nROM配置建议:")
    print(f"1. 深度: 2048")
    print(f"2. 宽度: 32位")
    print(f"3. 数据格式: 选择HEX或BIN")
    print(f"4. 确保文件路径正确")
    print(f"5. 文件编码: 纯文本(UTF-8或ASCII)")

if __name__ == "__main__":
    file_path = r"E:\my_FPGA\LMLP\7_20251108\mlp_weights_q10_22.dat"
    
    # 分析原始文件
    if analyze_weight_file(file_path):
        # 转换为十六进制格式
        hex_output = file_path.replace('.dat', '_hex.dat')
        convert_to_hex_format(file_path, hex_output)
        
        # 转换为二进制格式  
        bin_output = file_path.replace('.dat', '_bin.dat')
        convert_to_bin_format(file_path, bin_output)
        
        # 检查ROM配置
        check_rom_configuration()
        
        print(f"\n✅ 建议:")
        print(f"使用 {hex_output} 或 {bin_output} 作为ROM初始化文件")
        print(f"在IP配置中选择对应的格式(HEX或BIN)")