# 基于FPGA的MLP波形识别与双CORDIC数字锁相环系统

## 项目简介

本项目是一个完整的FPGA实现方案，集成了多层感知机(MLP)波形分类器与高性能双CORDIC数字锁相环(DCORDIC-PLL)，专为Pango Design Suite (PDS) 2022 SP6.1开发环境优化，并且很容易地移植到其他平台。系统可实时识别正弦波、方波、三角波等信号类型，并实现高精度频率跟踪与波形重建。

## 关键词
FPGA, MLP, PLL, CORDIC, 波形识别, Pango

## 核心特性

- **MLP波形识别**: 64输入-32-16-4结构的定点量化神经网络，支持4类波形实时分类4
- **微弱信号滤波与放大**:当前版本使用高速16bitADC，处理20mV以下的信号，滤波后放大从而重建信号
- **双CORDIC PLL**: 采用双CORDIC架构的32位高精度数字锁相环，具有12级自适应滤波系数
- **混合精度设计**: Q10.22定点数格式确保计算精度与资源效率的平衡
- **完整流水线**: 从ADC采样→PLL跟踪→MLP推理→UART输出的全硬件流水线
- **跨时钟域设计**: 支持27MHz/100MHz多时钟域协同工作

## 项目结构

```
├── verilog_source/          # FPGA源代码
│   ├── lmlp_top.v          # MLP顶层模块(含ADC接口与UART输出)
│   ├── mlp_classifier.v    # MLP权重/偏置查找表(LUT)
│   ├── sample.v            # 自适应采样模块
│   ├── pll_top.v           # PLL顶层控制模块
│   ├── pll_cordic.v        # 双CORDIC PLL核心
│   ├── seq_cordic.v        # 向量旋转CORDIC运算单元
│   ├── seq_polar.v         # 极坐标转换CORDIC单元
│   ├── uart_top.v          # UART接口封装
│   ├── uart_tx.v           # UART发送状态机
│   ├── top.v               # 系统顶层模块
│   ├── triangle_wave.v     # 三角波DDS生成器
│   └── key_ctrl.v          # 按键消抖与状态机
├── testbench/              # 仿真测试平台
│   ├── tb_mlp.v            # MLP功能仿真
│   └── tb_pll.v            # PLL跟踪性能仿真
├── python/                 # Python工具链
│   ├── train_mlp.py        # MLP训练脚本(PyTorch)
│   ├── pctofpga.py   		# 量化与Verilog代码生成
│   └── ...   				# 量化后模型验证;模型效果测试
└── docs/                   # 技术文档
    └── ReadMe.md     		# 项目介绍与使用指南
```

## 核心模块说明

### 1. **lmlp_top.v**: MLP推理引擎
- **功能**: 64点ADC采样→Q10.22归一化→三层前向传播→Softmax分类
- **关键参数**:
  ```verilog
  localparam INPUT_SIZE = 64;
  localparam HIDDEN_LAYER_1_SIZE = 32;
  localparam HIDDEN_LAYER_2_SIZE = 16;
  localparam OUTPUT_SIZE = 4;
  localparam Q_FRAC_BITS = 22;  // Q10.22格式
  ```
- **性能**: 135MHz时钟下推理延迟<1ms，支持UART实时输出分类结果与置信度

### 2. **pll_cordic.v**: 双CORDIC PLL核心
- **架构**: 
  - CORDIC乘法器: 执行信号与本地振荡器混频
  - IIR滤波器: 可配置log_alpha系数(3-14级)
  - CORDIC鉴相器: 极坐标转换提取相位误差
- **跟踪精度**: 32位相位分辨率，支持1Hz-1MHz信号跟踪

### 3. **sample.v**: 智能采样模块
- **算法**: 改进型过零检测 + 滤波
- **特性**: 
  - 自动偏置校准(基于IIR输出最大/最小值估计)
  - 自适应采样间隔 = 周期长度/64
  - 阈值: ZERO_THRESHOLD = 7680 (30 LSB)

## 定点量化方案

### Q10.22格式定义
```python
# Python量化示例
Q_SCALE = 1 << 22  # 2^22
def float_to_fixed(x):
    return int(round(x * Q_SCALE))
```

### MLP参数范围
- **权重**: `int32` (Q10.22)
- **偏置**: `int32` (Q10.22)  
- **激活值**: 每层后做饱和处理(saturate_64_to_32)

## 仿真与验证
略

## Python工具链使用
略

##使用指南与可能出现的问题：

###1.修改输入adc的比特数，调整信号的大小或者放大的倍数，尽量使得信号的幅值在20000LSB左右（以16bitADC为例）
###2.根据信号的比特数，调整MLP的ADC预处理部分中归一化系数（详情参照注释）
###3.可能需要重新训练AI：当前python训练的模型，使用的训练数据为高噪声信号，用户可以根据需要微调重新训练
###4.当前存在未包含的IP:（1）PLL(系统27M，生成100M,用于ADC采样，并且作为pll与mlp的工作时钟)；（2）DRAM（存储20k的模型权重参数，dat格式，python根据pth文件生成）
###5.关于pll需要注意：采样时钟小于PLL工作时钟的1/50，因为处理过程使用了大量的时序逻辑，CORDIC迭代较多（引自https://github.com/ZipCPU/dpll ，用户可以自行验证是否正确）

## 开源许可

本项目采用 **GPL v3** 许可证，核心CORDIC算法源自[ZipCPU/dpll](https://github.com/ZipCPU/dpll)项目，遵循GPL协议要求。

## 引用说明

欢迎您在学术或商业项目中使用本代码。

## 更新日志

- **V5.0**: 优化跨时钟域同步，修复UART打印竞争冒险
- **V4.0**: 改进BatchNorm实现，添加置信度计算
- **V3.0**: 修正信号偏置处理，提升采样精度
- **V2.0**: 实现12级自适应PLL系数，支持100kHz信号跟踪
- **V1.0**: 基础版本，支持1k-10kHz信号识别
- **V0.0**: Demo版本，仿真通过