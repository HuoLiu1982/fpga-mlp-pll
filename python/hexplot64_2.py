import matplotlib.pyplot as plt
import numpy as np

# 从UART接收到的ADC数据字符串（连续的十六进制字符）
# 注意：现在是16位数据，每4个字符代表一个16位的采样值
uart_adc_data = "59D85345483A425E3DC63A5634E92FF32AFD244820251F161BD91F3D1AF920C9201A1EBA216326972A112DF735953F2A456045104CD156F35D7F63FE6A2470D97CED823D8A5B8CA294E496589959A789A7F9A8EEA946AA1AA7E5A354A34EA3EFA0A49CD59A7094788CF982EA7DB57BDB75C56B9B61685FB156FD4D6845AD4191"

# 将连续的十六进制字符串转换为整数列表
# 每4个字符代表一个十六进制的16位数据
decimal_values = []
for i in range(0, len(uart_adc_data), 4):  # 修改：步长改为4
    hex_word = uart_adc_data[i:i+4]  # 修改：每次取4个字符
    decimal_values.append(int(hex_word, 16))  # 修改：解析为16位整数

print(f"解析出 {len(decimal_values)} 个数据点")
print(f"数据范围: {min(decimal_values)} - {max(decimal_values)}")

# 绘图
plt.figure(figsize=(14, 6))
plt.plot(decimal_values, 'b-', linewidth=1.5, label='ADC采样值')
plt.title("ADC采样数据波形 (16位)")
plt.xlabel("采样点索引")
plt.ylabel("幅值 (0-65535)")  # 修改：y轴范围改为16位
plt.grid(True, alpha=0.3)
plt.legend()
# 添加一些统计信息
plt.figtext(0.02, 0.02, f"数据点数: {len(decimal_values)}\n最大值: {max(decimal_values)}\n最小值: {min(decimal_values)}\n平均值: {np.mean(decimal_values):.2f}", 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.show()

# 可选：打印前20个数据点用于验证
print("\n前20个数据点:")
for i in range(min(20, len(decimal_values))):
    print(f"索引 {i}: 十六进制 0x{uart_adc_data[i*4:i*4+4]} -> 十进制 {decimal_values[i]}")  # 修改：索引步长改为4