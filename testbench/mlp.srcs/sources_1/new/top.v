`timescale 1ns / 1ps

module top (
    input  wire        clk_27M,
    input  wire        clk_135M,           // 改为输入，由外部提供
    input  wire [7:0]  ad_data_in,
    input  wire        rst_n,
    input  wire [31:0] rom_data_in,        // ROM数据输入，替换IP核
    output wire        ad_clk,
    output wire        uart_tx,
    output reg  [10:0] rom_addr,       // ROM地址输出，用于仿真。修改
    output reg  [1:0]  mlp_prediction_out, // MLP预测结果输出
    output reg signed [31:0] mlp_confidence_out, // MLP置信度输出
    output reg         mlp_done_out        // MLP完成信号输出
);

/* -------------------------------------------------
 * 1. 1.5 MHz 采样时钟
 * -------------------------------------------------*/
reg        clk_3M;
reg [4:0]  clk_cnt;
always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        clk_cnt <= 0;
        clk_3M  <= 0;
        $display("[CLK_GEN] 时钟生成器复位");
    end else if (clk_cnt == 8) begin
        clk_cnt <= 0;
        clk_3M  <= ~clk_3M;
    end else
        clk_cnt <= clk_cnt + 1'd1;
end
assign ad_clk = clk_3M;

reg pwok;
always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        pwok <= 0;
        $display("[PWOK] 上电复位");
    end else begin
        pwok <= 1; // 复位后上电完成
    end
end

/* -------------------------------------------------
 * 2. ADC采样和归一化处理 - 符号安全版本
 * -------------------------------------------------*/
reg [10:0] addr;
reg        sample_done;
reg [7:0]  ad_buffer [0:63];
reg signed [31:0] normalized_data [0:63];

// Q10.22格式参数
localparam Q_FRAC_BITS = 22;
localparam Q_SCALE = 1 << Q_FRAC_BITS;

// ADC归一化系数：1/255 in Q10.22 = 2^22 / 255
localparam NORM_COEFF = 32'h00004040;

// ADC采样状态机
reg [1:0] adc_state;
localparam ADC_IDLE = 2'd0;
localparam ADC_SAMPLING = 2'd1;
localparam ADC_NORMALIZING = 2'd2;
localparam ADC_DONE = 2'd3;

reg [5:0] norm_addr;
reg [39:0] temp_product;  // 40位中间结果
reg signed [31:0] final_value;

// ADC采样和归一化处理状态机
always @(posedge clk_3M or negedge rst_n) begin
    if (!rst_n) begin
        adc_state <= ADC_IDLE;
        addr <= 0;
        norm_addr <= 0;
        sample_done <= 0;
        $display("[ADC] ADC状态机复位");
    end else begin
        case(adc_state)
            ADC_IDLE: begin
                if (!sample_done) begin
                    adc_state <= ADC_SAMPLING;
                    addr <= 0;
                    $display("[ADC] 开始采样，进入ADC_SAMPLING状态");
                end
            end
            
            ADC_SAMPLING: begin
                ad_buffer[addr] <= ad_data_in;
                if (addr == 63) begin
                    adc_state <= ADC_NORMALIZING;
                    addr <= 0;
                    norm_addr <= 0;
                    $display("[ADC] 采样完成，进入ADC_NORMALIZING状态");
                end else begin
                    addr <= addr + 1'd1;
                end
            end
            
            ADC_NORMALIZING: begin
                // 无符号乘法
                temp_product = ad_buffer[norm_addr] * NORM_COEFF;
                
                // 确保结果在正确的范围内并转换为有符号
                if (temp_product[39:32] != 8'b0) begin
                    // 如果高位不为0，说明溢出，使用最大值
                    final_value = 32'h3FFFFFF;  // 接近1.0的Q10.22值
                    $display("[ADC_NORMALIZING] 警告：地址=%d 数据溢出，使用最大值", norm_addr);
                end else begin
                    // 正常情况，取低32位并确保是正数
                    final_value = {1'b0, temp_product[30:0]};  // 确保符号位为0
                end
                
                normalized_data[norm_addr] <= final_value;
                
                // 调试输出
                $display("[ADC_NORMALIZING] 地址=%d, ADC原始=%h, 临时乘积=%h, 归一化结果=%h", 
                         norm_addr, ad_buffer[norm_addr], temp_product, final_value);
                
                if (norm_addr == 63) begin
                    adc_state <= ADC_DONE;
                    sample_done <= 1;
                    $display("[ADC] 归一化完成，进入ADC_DONE状态");
                end else begin
                    norm_addr <= norm_addr + 1'd1;
                end
            end
            
            ADC_DONE: begin
                // 保持完成状态
                $display("[ADC_DONE] ADC处理完成，等待MLP推理");
            end
            
            default: begin
                adc_state <= ADC_IDLE;
                $display("[ADC] 错误：进入未知状态，返回IDLE");
            end
        endcase
    end
end

/* -------------------------------------------------
 * 4. MLP推理引擎（多周期ROM读取和乘法）
 * -------------------------------------------------*/
// 网络配置参数
localparam INPUT_SIZE = 64;
localparam HIDDEN_LAYER_1_SIZE = 32;
localparam HIDDEN_LAYER_2_SIZE = 16;
localparam OUTPUT_SIZE = 4;
integer i, j, k;
// 推理状态机
reg [3:0] mlp_state;
reg [10:0] mlp_counter;
reg signed [63:0] mlp_mac_accumulator;  // 改为64位防止溢出
reg [4:0] mlp_neuron_idx;
reg [5:0] mlp_input_idx;

// 添加流水线控制寄存器
reg pipeline_busy;

// 中间结果
reg signed [31:0] mlp_layer1_out [0:31];
reg signed [31:0] mlp_layer2_out [0:15];
reg signed [31:0] mlp_output_out [0:3];
reg [31:0] current_scale, current_shift;
reg signed [31:0] temp_relu;
reg signed [63:0] temp_mac;  // 改为64位
reg signed [31:0] temp_bn;
reg signed [31:0] temp_bias;
// 推理结果
reg [1:0] mlp_prediction;
reg [31:0] mlp_confidence;
reg        mlp_done;

// 状态定义
localparam MLP_IDLE = 4'd0;
localparam MLP_LAYER1_MAC = 4'd1;
localparam MLP_LAYER1_BIAS = 4'd2;
localparam MLP_LAYER1_BN = 4'd3;
localparam MLP_LAYER1_RELU = 4'd4;
localparam MLP_LAYER2_MAC = 4'd5;
localparam MLP_LAYER2_BIAS = 4'd6;
localparam MLP_LAYER2_BN = 4'd7;
localparam MLP_LAYER2_RELU = 4'd8;
localparam MLP_OUTPUT_MAC = 4'd9;
localparam MLP_OUTPUT_BIAS = 4'd10;
localparam MLP_FIND_MAX = 4'd11;
localparam MLP_DONE = 4'd12;


// ROM接口（替换IP核）
//reg [10:0] rom_addr;//修改
wire signed [31:0] rom_data;  // 直接使用输入数据
assign rom_data = rom_data_in;
// LUT接口
wire signed[31:0] layer1_bias_data;
wire signed[31:0] layer2_weight_data;
wire signed[31:0] layer2_bias_data;
wire signed[31:0] output_weight_data;
wire signed[31:0] output_bias_data;
wire signed[31:0] bn1_scale_data;
wire signed[31:0] bn1_shift_data;
wire signed[31:0] bn2_scale_data;
wire signed[31:0] bn2_shift_data;

// 实例化LUT模块
/*
layer1_bias_lut u_layer1_bias(.addr(rom_addr[4:0]), .data(layer1_bias_data));
layer2_weight_lut u_layer2_weight(.addr(rom_addr[9:0]), .data(layer2_weight_data));
layer2_bias_lut u_layer2_bias(.addr(rom_addr[3:0]), .data(layer2_bias_data));
output_weight_lut u_output_weight(.addr(rom_addr[5:0]), .data(output_weight_data));
output_bias_lut u_output_bias(.addr(rom_addr[1:0]), .data(output_bias_data));
bn1_scale_lut u_bn1_scale(.addr(rom_addr[4:0]), .data(bn1_scale_data));
bn1_shift_lut u_bn1_shift(.addr(rom_addr[4:0]), .data(bn1_shift_data));
bn2_scale_lut u_bn2_scale(.addr(rom_addr[3:0]), .data(bn2_scale_data));
bn2_shift_lut u_bn2_shift(.addr(rom_addr[3:0]), .data(bn2_shift_data));
*/
layer1_bias_lut u_layer1_bias(.addr(mlp_neuron_idx), .data(layer1_bias_data));
layer2_weight_lut u_layer2_weight(.addr(rom_addr[9:0]), .data(layer2_weight_data));
layer2_bias_lut u_layer2_bias(.addr(mlp_neuron_idx), .data(layer2_bias_data));
output_weight_lut u_output_weight(.addr(rom_addr[5:0]), .data(output_weight_data));
output_bias_lut u_output_bias(.addr(mlp_neuron_idx), .data(output_bias_data));
bn1_scale_lut u_bn1_scale(.addr(mlp_neuron_idx), .data(bn1_scale_data));
bn1_shift_lut u_bn1_shift(.addr(mlp_neuron_idx), .data(bn1_shift_data));
bn2_scale_lut u_bn2_scale(.addr(mlp_neuron_idx), .data(bn2_scale_data));
bn2_shift_lut u_bn2_shift(.addr(mlp_neuron_idx), .data(bn2_shift_data));
// ROM读取流水线控制
reg [1:0] rom_read_state;
reg signed [31:0] rom_data_reg;
reg signed [31:0] normalized_data_reg;
reg signed [31:0] layer1_out_reg;
reg        rom_read_valid;
reg signed [31:0] layer2_out_reg;
// 乘法器流水线
reg signed [63:0] multiply_result;
reg        multiply_valid;
reg [1:0]  multiply_state;

localparam ROM_READ_IDLE = 2'd0;
localparam ROM_READ_SETUP = 2'd1;
localparam ROM_READ_FETCH = 2'd2;

localparam MULTIPLY_IDLE = 2'd0;
localparam MULTIPLY_EXECUTE = 2'd1;

// 添加握手信号
reg multiply_busy;           // 乘法器忙碌标志
reg [15:0] pending_rom_addr; // 待处理的ROM地址


// ROM读取状态机 - 完全重写
always @(posedge clk_135M or negedge rst_n) begin
    if (!rst_n) begin
        rom_read_state <= ROM_READ_IDLE;
        rom_read_valid <= 0;
        rom_data_reg <= 0;
        normalized_data_reg <= 0;
        layer1_out_reg <= 0;
        layer2_out_reg <= 0;
        multiply_busy <= 0;
        pending_rom_addr <= 0;
        $display("[ROM_READ] ROM读取状态机复位");
    end else begin
        rom_read_valid <= 0;
        
        case(rom_read_state)
            ROM_READ_IDLE: begin
                if ((mlp_state == MLP_LAYER1_MAC || mlp_state == MLP_LAYER2_MAC || 
                     mlp_state == MLP_OUTPUT_MAC) && !multiply_busy) begin
                    // 锁定当前地址并请求读取
                    pending_rom_addr <= rom_addr;
                    rom_read_state <= ROM_READ_SETUP;
                    $display("[ROM_READ_IDLE] 请求ROM读取，地址=%d，MLP状态=%d", rom_addr, mlp_state);
                end
            end
            
            ROM_READ_SETUP: begin
                // 确保ROM地址稳定
                if (rom_addr == pending_rom_addr) begin
                    // 准备输入数据
                    if (mlp_state == MLP_LAYER1_MAC) begin
                        normalized_data_reg <= normalized_data[mlp_input_idx];
                        $display("[ROM_READ_SETUP] L1_MAC: 地址=%d, 归一化数据=%h", 
                                mlp_input_idx, normalized_data[mlp_input_idx]);
                    end else if (mlp_state == MLP_LAYER2_MAC) begin
                        layer1_out_reg <= mlp_layer1_out[mlp_input_idx];
                        $display("[ROM_READ_SETUP] L2_MAC: 地址=%d, 层1输出=%h", 
                                mlp_input_idx, mlp_layer1_out[mlp_input_idx]);
                    end else if (mlp_state == MLP_OUTPUT_MAC) begin
                        layer2_out_reg <= mlp_layer2_out[mlp_input_idx];
                        $display("[ROM_READ_SETUP] OUT_MAC: 地址=%d, 层2输出=%h", 
                                mlp_input_idx, mlp_layer2_out[mlp_input_idx]);
                    end
                    rom_read_state <= ROM_READ_FETCH;
                end else begin
                    // 地址已改变，重新开始
                    $display("[ROM_READ_SETUP] 地址改变，重新开始");
                    rom_read_state <= ROM_READ_IDLE;
                end
            end
            
            ROM_READ_FETCH: begin
                // 获取ROM数据并标记为有效
                rom_data_reg <= rom_data;
                rom_read_valid <= 1;
                multiply_busy <= 1; // 标记乘法器开始工作
                rom_read_state <= ROM_READ_IDLE;
                $display("[ROM_READ_FETCH] 获取ROM数据=%h，地址=%d", rom_data, pending_rom_addr);
            end
        endcase
        
        // 乘法完成时清除忙碌标志
        if (multiply_valid) begin
            multiply_busy <= 0;
            $display("[ROM_READ] 乘法完成，清除忙碌标志");
        end
    end
end

// 乘法器状态机
always @(posedge clk_135M or negedge rst_n) begin
    if (!rst_n) begin
        multiply_result <= 0;
        multiply_valid <= 0;
        multiply_state <= MULTIPLY_IDLE;
        $display("[MULTIPLY] 乘法器流水线复位");
    end else begin
        multiply_valid <= 0;
        
        case(multiply_state)
            MULTIPLY_IDLE: begin
                if (rom_read_valid && !multiply_valid) begin
                    multiply_state <= MULTIPLY_EXECUTE;
                    
                    // 执行乘法
                    if (mlp_state == MLP_LAYER1_MAC) begin
                        multiply_result <= rom_data_reg * normalized_data_reg;
                        $display("[MULTIPLY] L1_MAC: ROM数据=%h * 归一化数据=%h", 
                                rom_data_reg, normalized_data_reg);
                    end else if (mlp_state == MLP_LAYER2_MAC) begin
                        multiply_result <= layer2_weight_data * layer1_out_reg;
                        $display("[MULTIPLY] L2_MAC: 权重=%h * 层1输出=%h", 
                                layer2_weight_data, layer1_out_reg);
                    end else if (mlp_state == MLP_OUTPUT_MAC) begin
                        multiply_result <= output_weight_data * layer2_out_reg;
                        $display("[MULTIPLY] OUT_MAC: 权重=%h * 层2输出=%h", 
                                output_weight_data, layer2_out_reg);
                    end
                end
            end
            
            MULTIPLY_EXECUTE: begin
                multiply_valid <= 1;
                multiply_state <= MULTIPLY_IDLE;
                $display("[MULTIPLY] 乘法完成，结果=%h", multiply_result);
            end
        endcase
    end
end

// 提取Q10.22结果
function signed [31:0] extract_q10_22;
    input signed [63:0] product;
    reg signed [31:0] result;
    begin
        // 添加偏移量进行舍入
        product = product + (1 << (Q_FRAC_BITS - 1));
        result = product >>> Q_FRAC_BITS;  // 右移22位得到Q10.22
        $display("[EXTRACT_Q10_22] 输入乘积=%h, 输出结果=%h", product, result);
        extract_q10_22 = result;
    end
endfunction

// 64位到32位的有符号饱和处理
function signed [31:0] saturate_64_to_32;
    input signed [63:0] value;
    begin
        if (value > $signed(64'sh000000007FFFFFFF)) begin
            // 大于32位有符号最大值，饱和到最大值
            saturate_64_to_32 = 32'sh7FFFFFFF;
            $display("[SATURATE] 64位值=%h 超过32位正最大值，饱和到7FFFFFFF", value);
        end else if (value < $signed(64'shFFFFFFFF80000000)) begin
            // 小于32位有符号最小值，饱和到最小值
            saturate_64_to_32 = 32'sh80000000;
            $display("[SATURATE] 64位值=%h 超过32位负最小值，饱和到80000000", value);
        end else begin
            // 在32位范围内，直接截断
            saturate_64_to_32 = value[31:0];
            $display("[SATURATE] 64位值=%h 在32位范围内，截断为=%h", value, value[31:0]);
        end
    end
endfunction

// 修正的ReLU激活函数
function signed [31:0] relu_fixed;
    input signed [31:0] x;
    begin
        if (x < 0) begin
            $display("[RELU] 输入=%h (负数), 输出=0", x);
            relu_fixed = 0;
        end else begin
            $display("[RELU] 输入=%h (正数), 输出保持不变", x);
            relu_fixed = x;
        end      
    end
endfunction

// 改进的BatchNorm实现
function signed [31:0] batchnorm_simple;
    input signed [31:0] x;
    input signed [31:0] scale;
    input signed [31:0] shift;
    reg signed [63:0] scaled;
    reg signed [63:0] rounded;
    begin
        scaled = $signed(x) * $signed(scale);
        // 更好的舍入处理
        rounded = scaled + (1 << (Q_FRAC_BITS - 1));
        batchnorm_simple = (rounded >>> Q_FRAC_BITS) + shift;
        
        $display("[IMPROVED_BN] x=%h, scale=%h, scaled=%h, result=%h", 
                 x, scale, scaled, batchnorm_simple);
    end
endfunction

// 跨时钟域同步：检测sample_done的上升沿（3MHz -> 135MHz）
reg sample_done_sync1, sample_done_sync2, sample_done_sync3;
wire sample_done_rising;

always @(posedge clk_135M or negedge rst_n) begin
    if (!rst_n) begin
        sample_done_sync1 <= 0;
        sample_done_sync2 <= 0;
        sample_done_sync3 <= 0;
        $display("[SYNC] 跨时钟域同步复位");
    end else begin
        sample_done_sync1 <= sample_done;
        sample_done_sync2 <= sample_done_sync1;
        sample_done_sync3 <= sample_done_sync2;
        if (sample_done_sync2 && !sample_done_sync3) begin
            $display("[SYNC] 检测到sample_done上升沿");
        end
    end
end
assign sample_done_rising = sample_done_sync2 && !sample_done_sync3;

// MLP推理状态机（135MHz时钟）
always @(posedge clk_135M or negedge rst_n) begin
    if (!rst_n) begin
        mlp_state <= MLP_IDLE;
        mlp_counter <= 0;
        mlp_done <= 0;
        mlp_prediction <= 0;
        mlp_confidence <= 0;
        rom_addr <= 0;
        mlp_mac_accumulator <= 0;
        mlp_neuron_idx <= 0;
        mlp_input_idx <= 0;
        pipeline_busy <= 0;
        // 初始化数组
        for (i = 0; i < 32; i = i + 1) mlp_layer1_out[i] <= 0;
        for (j = 0; j < 16; j = j + 1) mlp_layer2_out[j] <= 0;
        for (k = 0; k < 4; k = k + 1) mlp_output_out[k] <= 0;
        
        $display("[MLP] MLP状态机复位");
    end else begin
        case(mlp_state)
            MLP_IDLE: begin
                if (sample_done_rising) begin
                    mlp_state <= MLP_LAYER1_MAC;
                    mlp_counter <= 0;
                    mlp_neuron_idx <= 0;
                    mlp_input_idx <= 0;
                    rom_addr <= 0;
                    mlp_mac_accumulator <= 0;
                    $display("[MLP] 开始推理，进入MLP_LAYER1_MAC状态");
                end
                mlp_done <= 0;
            end
            
            MLP_LAYER1_MAC: begin
                // 提前设置ROM地址，确保在ROM读取开始前地址就稳定
                if (!pipeline_busy && rom_read_state == ROM_READ_IDLE) begin
                    rom_addr <= (mlp_neuron_idx * INPUT_SIZE) + mlp_input_idx;
                    pipeline_busy <= 1;  // 标记流水线忙碌
                    $display("[MLP_LAYER1_MAC] 设置ROM地址=%d", (mlp_neuron_idx * INPUT_SIZE) + mlp_input_idx);
                end
                if (mlp_counter < 2048) begin                   
                    if (multiply_valid) begin
                        // 累加乘法结果 - 使用64位累加器
                        temp_mac = mlp_mac_accumulator + $signed(extract_q10_22(multiply_result));
                        //temp_mac = mlp_mac_accumulator + $signed(multiply_result);
                        mlp_mac_accumulator <= temp_mac;
                        
                        $display("[MLP_LAYER1_MAC] MAC更新: 计数器=%d, 神经元=%d, 输入=%d, 旧累加器=%h, 新累加器=%h", 
                                 mlp_counter, mlp_neuron_idx,mlp_input_idx, mlp_mac_accumulator, temp_mac);
                        
                        mlp_input_idx <= mlp_input_idx + 1;
                        pipeline_busy <= 0;  // 清除忙碌标志，允许启动下一次读取
                        if (mlp_input_idx == 63) begin
                            // 完成当前神经元的全部输入
                            mlp_state <= MLP_LAYER1_BIAS;
                            mlp_input_idx <= 0;
                            $display("[MLP] 层1 MAC完成，进入BIAS状态");
                        end else begin
                            // 准备下一个输入
                            mlp_counter <= mlp_counter + 1;
                            $display("[MLP_LAYER1_MAC] 准备下一个输入，新计数器=%d", mlp_counter + 1);
                        end
                    end
                 end else begin
                        // 防止卡死
                        mlp_state <= MLP_LAYER1_BIAS;
                        mlp_input_idx <= 0;
                        pipeline_busy <= 0;  // 清除忙碌标志
                        $display("[MLP] 警告：计数器超限，强制进入BIAS状态");
                end
            end
            MLP_LAYER1_BIAS: begin
                // 添加偏置 - 使用饱和处理将64位转换为32位
                rom_addr <= mlp_neuron_idx;
                
                // 将偏置符号扩展为64位，然后与累加器相加，最后进行饱和处理
                temp_mac = mlp_mac_accumulator + {{32{layer1_bias_data[31]}}, layer1_bias_data};
                temp_bias = saturate_64_to_32(temp_mac);
                mlp_layer1_out[mlp_neuron_idx] <= temp_bias;
                mlp_mac_accumulator <= 0;
                
                $display("[MLP_LAYER1_BIAS] 神经元=%d, MAC结果=%h, 偏置=%h, 最终输出=%h", 
                         mlp_neuron_idx, mlp_mac_accumulator, layer1_bias_data, mlp_layer1_out[mlp_neuron_idx]);
                
                // 设置下一个神经元的起始地址
                mlp_counter <= (mlp_neuron_idx + 1) * 64;
                mlp_input_idx <= 0;
                pipeline_busy <= 0;
                mlp_mac_accumulator <= 0;
                if (mlp_neuron_idx == 31) begin
                    // 所有神经元处理完成
                    mlp_state <= MLP_LAYER1_BN;
                            mlp_neuron_idx <= 0;
                            mlp_input_idx <= 0;
                            pipeline_busy <= 0;  // 清除忙碌标志
                    rom_addr <= 0;
                    $display("[MLP] 层1偏置完成，进入BN状态");
                end else begin
                    // 处理下一个神经元
                    mlp_neuron_idx <= mlp_neuron_idx + 1;
                    mlp_state <= MLP_LAYER1_MAC;
                    $display("[MLP] 处理下一个神经元: %d", mlp_neuron_idx + 1);
                end
            end
            
            MLP_LAYER1_BN: begin
                // 设置BN参数地址
                rom_addr <= mlp_neuron_idx;
                
                // 使用寄存器暂存BN参数，避免组合逻辑延迟               
                current_scale = bn1_scale_data;
                current_shift = bn1_shift_data;
                
                $display("[MLP_LAYER1_BN] 神经元=%d, 输入=%h, scale=%h, shift=%h", 
                         mlp_neuron_idx, mlp_layer1_out[mlp_neuron_idx], current_scale, current_shift);
                
                temp_bn = batchnorm_simple(
                    mlp_layer1_out[mlp_neuron_idx],
                    current_scale,
                    current_shift
                );
                mlp_layer1_out[mlp_neuron_idx] <= temp_bn;
                $display("[MLP_LAYER1_BN] BN后输出=%h", temp_bn);
                
                mlp_neuron_idx <= mlp_neuron_idx + 1;
                
                if (mlp_neuron_idx == 31) begin
                    mlp_state <= MLP_LAYER1_RELU;
                    mlp_neuron_idx <= 0;
                    rom_addr <= 0;
                    $display("[MLP] 层1 BN完成，进入RELU状态");
                end
            end
            
            MLP_LAYER1_RELU: begin
                // ReLU处理：一个周期完成所有神经元
                for (i = 0; i < 32; i = i + 1) begin
                    temp_relu = relu_fixed(mlp_layer1_out[i]);
                    mlp_layer1_out[i] <= temp_relu;
                    $display("[MLP_LAYER1_RELU] 神经元=%d, ReLU后=%h", i, temp_relu);
                end
                mlp_state <= MLP_LAYER2_MAC;
                mlp_neuron_idx <= 0;
                mlp_input_idx <= 0;
                mlp_counter <= 0;
                rom_addr <= 0;
                $display("[MLP] 层1 RELU完成，进入层2 MAC状态");
            end
            
            MLP_LAYER2_MAC: begin
                if (mlp_counter < 512) begin
                    if (!pipeline_busy && rom_read_state == ROM_READ_IDLE) begin
                        rom_addr <= mlp_counter;
                        pipeline_busy <= 1;
                        $display("[MLP_LAYER2_MAC] 启动ROM读取，地址=%d", mlp_counter);
                    end
                    
                    if (multiply_valid) begin
                        // 累加乘法结果 - 使用64位累加器
                        temp_mac = mlp_mac_accumulator + $signed(extract_q10_22(multiply_result));
                        //temp_mac = mlp_mac_accumulator + $signed(multiply_result);
                        mlp_mac_accumulator <= temp_mac;
                        
                        $display("[MLP_LAYER2_MAC] MAC更新: 计数器=%d, 神经元=%d,输入=%d, 旧累加器=%h, 新累加器=%h", 
                                 mlp_counter, mlp_neuron_idx,mlp_input_idx, mlp_mac_accumulator, temp_mac);
                        
                        mlp_input_idx <= mlp_input_idx + 1;
                        
                        if (mlp_input_idx == 31) begin
                            mlp_state <= MLP_LAYER2_BIAS;
                            mlp_input_idx <= 0;
                            pipeline_busy <= 0;  // 清除忙碌标志
                            $display("[MLP] 层2 MAC完成，进入BIAS状态");
                        end else begin
                            mlp_counter <= mlp_counter + 1;
                            pipeline_busy <= 0;
                            $display("[MLP_LAYER2_MAC] 准备下一个输入，新计数器=%d", mlp_counter + 1);
                        end
                    end
                end else begin
                    mlp_state <= MLP_LAYER2_BIAS;
                            mlp_input_idx <= 0;
                            pipeline_busy <= 0;  // 清除忙碌标志
                    $display("[MLP] 警告：计数器超限，强制进入BIAS状态");
                end
            end           
            MLP_LAYER2_BIAS: begin
                rom_addr <= mlp_neuron_idx;
                // 将偏置符号扩展为64位，然后与累加器相加，最后进行饱和处理
                temp_mac = mlp_mac_accumulator + {{32{layer2_bias_data[31]}}, layer2_bias_data};
                temp_bias = saturate_64_to_32(temp_mac);
                mlp_layer2_out[mlp_neuron_idx] <= temp_bias;
                mlp_mac_accumulator <= 0;
                
                $display("[MLP_LAYER2_BIAS] 神经元=%d, MAC结果=%h, 偏置=%h, 最终输出=%h", 
                         mlp_neuron_idx, mlp_mac_accumulator, layer2_bias_data, mlp_layer2_out[mlp_neuron_idx]);
                
                // 设置下一个神经元的起始地址
                mlp_counter <= (mlp_neuron_idx + 1) * 32;
                mlp_input_idx <= 0;
                pipeline_busy <= 0;
                mlp_mac_accumulator <= 0;
                if (mlp_neuron_idx == 15) begin
                    mlp_state <= MLP_LAYER2_BN;
                    mlp_neuron_idx <= 0;
                    rom_addr <= 0;
                    $display("[MLP] 层2偏置完成，进入BN状态");
                end else begin
                    mlp_neuron_idx <= mlp_neuron_idx + 1;
                    mlp_state <= MLP_LAYER2_MAC;
                    $display("[MLP] 处理下一个神经元: %d", mlp_neuron_idx + 1);
                end
            end
            
            MLP_LAYER2_BN: begin
                // 修正地址生成
                rom_addr <= mlp_neuron_idx[3:0];  // 直接使用4位地址
                current_scale = bn2_scale_data;
                current_shift = bn2_shift_data;
                
                $display("[MLP_LAYER2_BN] 神经元=%d, 输入=%h, scale=%h, shift=%h", 
                         mlp_neuron_idx, mlp_layer2_out[mlp_neuron_idx], current_scale, current_shift);
                
                temp_bn = batchnorm_simple(
                    mlp_layer2_out[mlp_neuron_idx],
                    current_scale,
                    current_shift
                );
                mlp_layer2_out[mlp_neuron_idx] <= temp_bn;
                $display("[MLP_LAYER2_BN] BN后输出=%h", temp_bn);
                
                mlp_neuron_idx <= mlp_neuron_idx + 1;
                
                if (mlp_neuron_idx == 15) begin
                    mlp_state <= MLP_LAYER2_RELU;
                    mlp_neuron_idx <= 0;
                    rom_addr <= 0;
                    $display("[MLP] 层2 BN完成，进入RELU状态");
                end
            end
            
            MLP_LAYER2_RELU: begin
                // ReLU处理：一个周期完成所有神经元
                for (i = 0; i < 16; i = i + 1) begin
                    temp_relu = relu_fixed(mlp_layer2_out[i]);
                    mlp_layer2_out[i] <= temp_relu;
                    $display("[MLP_LAYER2_RELU] 神经元=%d, ReLU后=%h", i, temp_relu);
                end
                mlp_state <= MLP_OUTPUT_MAC;
                mlp_neuron_idx <= 0;
                mlp_input_idx <= 0;
                mlp_counter <= 0;
                rom_addr <= 0;
                $display("[MLP] 层2 RELU完成，进入输出层MAC状态");
            end
            
            MLP_OUTPUT_MAC: begin
                if (mlp_counter < 64) begin
                    if (rom_read_state == ROM_READ_IDLE) begin
                        rom_addr <= mlp_counter;
                    end
                    
                    if (multiply_valid) begin
                        // 累加乘法结果 - 使用64位累加器
                        temp_mac = mlp_mac_accumulator + $signed(extract_q10_22(multiply_result));
                        //temp_mac = mlp_mac_accumulator + $signed(multiply_result);
                        mlp_mac_accumulator <= temp_mac;
                        mlp_input_idx <= mlp_input_idx + 1;
                        mlp_counter <= mlp_counter + 1;
                        
                        $display("[MLP_OUTPUT_MAC] MAC更新: 计数器=%d, 神经元=%d,累加器=%h,新累加器=%h",
                            mlp_counter, mlp_neuron_idx, mlp_mac_accumulator,temp_mac);
                        
                        if (mlp_input_idx == 15) begin
                            mlp_state <= MLP_OUTPUT_BIAS;
                            mlp_input_idx <= 0;
                            $display("[MLP] 输出层 MAC完成，进入BIAS状态");
                        end else begin
                            rom_addr <= mlp_counter + 1;
                        end
                    end
                end else begin
                    // 防止卡死
                    mlp_state <= MLP_OUTPUT_BIAS;
                    $display("[MLP] 警告：计数器超限，强制进入BIAS状态");
                end
            end
            
            MLP_OUTPUT_BIAS: begin
                // 将偏置符号扩展为64位，然后与累加器相加，最后进行饱和处理
                temp_mac = mlp_mac_accumulator + {{32{output_bias_data[31]}}, output_bias_data};
                temp_bias = saturate_64_to_32(temp_mac);
                mlp_output_out[mlp_neuron_idx] <= temp_bias;
                mlp_mac_accumulator <= 0;
                
                $display("[MLP_OUTPUT_BIAS] 输出神经元=%d, MAC结果=%h, 偏置=%h, 最终输出=%h", 
                         mlp_neuron_idx, mlp_mac_accumulator, output_bias_data, mlp_output_out[mlp_neuron_idx]);
                
                mlp_neuron_idx <= mlp_neuron_idx + 1;
                
                if (mlp_neuron_idx == 3) begin
                    mlp_state <= MLP_FIND_MAX;
                    mlp_neuron_idx <= 0;
                    $display("[MLP] 输出层偏置完成，进入FIND_MAX状态");
                end else begin
                    mlp_state <= MLP_OUTPUT_MAC;
                    mlp_counter <= (mlp_neuron_idx + 1) * 16;
                    mlp_input_idx <= 0;
                    $display("[MLP] 处理下一个输出神经元: %d", mlp_neuron_idx + 1);
                end
            end
            
            MLP_FIND_MAX: begin
                // 找到最大值作为预测结果
                $display("[MLP_FIND_MAX] 输出值: [0]=%h, [1]=%h, [2]=%h, [3]=%h", 
                         mlp_output_out[0], mlp_output_out[1], mlp_output_out[2], mlp_output_out[3]);
                
                if (mlp_output_out[0] >= mlp_output_out[1] && 
                    mlp_output_out[0] >= mlp_output_out[2] && 
                    mlp_output_out[0] >= mlp_output_out[3]) begin
                    mlp_prediction <= 2'b00; // 正弦波
                    mlp_confidence <= mlp_output_out[0];
                    $display("[MLP_FIND_MAX] 预测结果: 正弦波, 置信度=%h", mlp_output_out[0]);
                end else if (mlp_output_out[1] >= mlp_output_out[0] && 
                           mlp_output_out[1] >= mlp_output_out[2] && 
                           mlp_output_out[1] >= mlp_output_out[3]) begin
                    mlp_prediction <= 2'b01; // 方波
                    mlp_confidence <= mlp_output_out[1];
                    $display("[MLP_FIND_MAX] 预测结果: 方波, 置信度=%h", mlp_output_out[1]);
                end else if (mlp_output_out[2] >= mlp_output_out[0] && 
                           mlp_output_out[2] >= mlp_output_out[1] && 
                           mlp_output_out[2] >= mlp_output_out[3]) begin
                    mlp_prediction <= 2'b10; // 三角波
                    mlp_confidence <= mlp_output_out[2];
                    $display("[MLP_FIND_MAX] 预测结果: 三角波, 置信度=%h", mlp_output_out[2]);
                end else begin
                    mlp_prediction <= 2'b11; // 其他
                    mlp_confidence <= mlp_output_out[3];
                    $display("[MLP_FIND_MAX] 预测结果: 其他, 置信度=%h", mlp_output_out[3]);
                end
                mlp_state <= MLP_DONE;
            end
            
            MLP_DONE: begin
                mlp_done <= 1;
                $display("[MLP_DONE] MLP推理完成! 预测=%b, 置信度=%h", mlp_prediction, mlp_confidence);
                // 保持状态直到新的采样开始
                if (!sample_done_sync2) begin
                    mlp_state <= MLP_IDLE;
                    mlp_done <= 0;
                    $display("[MLP] 返回IDLE状态，等待下一次推理");
                end
            end
            
            default: begin
                mlp_state <= MLP_IDLE;
                $display("[MLP] 错误：进入未知状态，返回IDLE");
            end
        endcase
    end
end

// 输出MLP结果用于仿真
always @(posedge clk_135M) begin
    mlp_prediction_out <= mlp_prediction;
    mlp_confidence_out <= mlp_confidence;
    mlp_done_out <= mlp_done;
end

/* -------------------------------------------------
 * 5. UART发送控制 - 简化为直接输出
 * -------------------------------------------------*/
// 简化UART，直接输出MLP结果
assign uart_tx = 1'b1;  // 保持高电平，避免干扰

endmodule