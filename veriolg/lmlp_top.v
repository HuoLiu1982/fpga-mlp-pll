`timescale 1ns / 1ps

module lmlp_top (
    input  wire        clk_27M,
    input  wire        clk_135M,//仿真测试135M,实际例化使用100M，与pll工作时钟以及adc时钟统一
    input  wire [15:0] ad_data_in,
    input  wire        rst_n,
    output reg [1:0]   mlp_prediction,
    output reg         mlp_done,
    output wire        uart_tx
);

/* -------------------------------------------------
 * 1MHz 采样时钟 - 修正版本
 * -------------------------------------------------*/
parameter DIVIDER = 100;
parameter CNT_MAX = DIVIDER/2 - 1;

reg clk_1M;
reg [7:0] clk_cnt;
reg clk_1M_prev;
reg pulse_ad;

always @(posedge clk_135M or negedge rst_n) begin
    if(!rst_n) begin
        clk_cnt <= 8'd0;
        clk_1M <= 1'b0;
        clk_1M_prev <= 1'b0;
        pulse_ad <= 1'b0;
    end else begin
        clk_1M_prev <= clk_1M;
        
        if(clk_cnt == CNT_MAX) begin
            clk_1M <= ~clk_1M;
            clk_cnt <= 8'd0;
        end else begin
            clk_cnt <= clk_cnt + 8'd1;
        end
        
        pulse_ad <= (clk_1M && !clk_1M_prev);
    end
end


/* -------------------------------------------------
 * 上电延时（no）
 * -------------------------------------------------*/
reg        pwok;

always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        pwok  <= 1;
    end
end

/* -------------------------------------------------
 * ADC采样和归一化处理
 * -------------------------------------------------*/
reg [10:0] addr;
reg        sample_done;
reg [15:0]  ad_buffer [0:63];
wire [15:0]  sample_data;
reg signed [31:0] normalized_data [0:63];

// Q10.22格式参数
localparam Q_FRAC_BITS = 22;
localparam Q_SCALE = 1 << Q_FRAC_BITS;

// ADC归一化系数：1/255 in Q10.22 = 2^22 / 255，改成1/2**16
localparam NORM_COEFF = 32'h00000040;

// ADC采样状态机
reg [1:0] adc_state;
localparam ADC_IDLE = 2'd0;
localparam ADC_SAMPLING = 2'd1;
localparam ADC_NORMALIZING = 2'd2;
localparam ADC_DONE = 2'd3;

reg [5:0] norm_addr;
reg [47:0] temp_product;  // 47位中间结果
reg signed [31:0] final_value;
integer i, j, k;
reg sample_en;
wire sam_done;
sample sample_uut(
    .clk_100M(clk_135M),
    .rst_n(rst_n),
    .enable(sample_en),
    .adc_dai_A(ad_data_in),
    .done(sam_done),
    .signal_tem(sample_data),//16位
    .signal_out()//8位
);

// ADC采样和归一化处理状态机
always @(posedge clk_135M or negedge rst_n) begin
    if (!rst_n) begin
        adc_state <= ADC_IDLE;
        addr <= 0;
        norm_addr <= 0;
        sample_done <= 0;
        sample_en <= 0;
        // 初始化缓冲区
        for (i = 0; i < 64; i = i + 1) begin
            ad_buffer[i] <= 0;
            normalized_data[i] <= 0;
        end
        $display("[ADC] ADC状态机复位");
    end else if (pwok) begin  // 只有上电完成后才运行
        case(adc_state)
            ADC_IDLE: begin
                if (!sample_done) begin
                    adc_state <= ADC_SAMPLING;
                    sample_en <= 1;
                    addr <= 0;
                    $display("[ADC] 开始采样，进入ADC_SAMPLING状态");
                end
            end
            
            ADC_SAMPLING: begin                        
                if (sam_done) begin
                    sample_en <= 0;
                    ad_buffer[addr] <= sample_data;
                    
                    if (addr == 63) begin
                        adc_state <= ADC_NORMALIZING;
                        addr <= 0;
                        norm_addr <= 0;
                    end else begin
                        addr <= addr + 1'd1;
                    end
                end
            end
            
            ADC_NORMALIZING: begin
                // 无符号乘法
                temp_product = ad_buffer[norm_addr] * NORM_COEFF;
                
                // 确保结果在正确的范围内并转换为有符号
                if (temp_product[47:32] != 16'b0) begin
                    // 如果高位不为0，说明溢出，使用最大值
                    final_value = 32'h3FFFFFF;  // 接近1.0的Q10.22值
                end else begin
                    // 正常情况，取低32位并确保是正数
                    final_value = {1'b0, temp_product[30:0]};  // 确保符号位为0
                end
                
                normalized_data[norm_addr] <= final_value;

                
                if (norm_addr == 63) begin
                    adc_state <= ADC_DONE;
                    sample_done <= 1;
                end else begin
                    norm_addr <= norm_addr + 1'd1;
                end
            end
            
            ADC_DONE: begin
                // 保持完成状态
            end
            
            default: begin
                adc_state <= ADC_IDLE;
                $display("[ADC] 错误：进入未知状态，返回IDLE");
            end
        endcase
    end else begin
        // 上电未完成，保持复位状态
        adc_state <= ADC_IDLE;
        sample_done <= 0;
    end
end
/* -------------------------------------------------
 * 5. MLP推理引擎（多周期ROM读取和乘法）
 * -------------------------------------------------*/
// 网络配置参数
localparam INPUT_SIZE = 64;
localparam HIDDEN_LAYER_1_SIZE = 32;
localparam HIDDEN_LAYER_2_SIZE = 16;
localparam OUTPUT_SIZE = 4;
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
reg [31:0] mlp_confidence;


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

// ROM接口（用于layer1权重）
reg [10:0] rom_addr;
wire signed[31:0] rom_data;
rom_weights u_rom_weights (
  .addr(rom_addr),          // input [10:0]
  .clk(clk_135M),           // input
  .rst(!rst_n),             // input
  .rd_data(rom_data)        // output [31:0]
);

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
    end else begin
        sample_done_sync1 <= sample_done;
        sample_done_sync2 <= sample_done_sync1;
        sample_done_sync3 <= sample_done_sync2;
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
/* -------------------------------------------------
 * 6. UART发送控制 - 扩展支持MLP结果打印
 * -------------------------------------------------*/
reg [10:0] uart_cnt;
reg [7:0]  uart_data;
reg        uart_start_send;
reg        uart_sending;
reg        mlp_done_reg;
reg [2:0]  uart_state;
wire       mlp_done_rising;
// UART发送状态机
localparam UART_IDLE = 3'd0;
localparam UART_SEND_ADC_HEADER = 3'd1;
localparam UART_SEND_ADC_DATA = 3'd2;
localparam UART_SEND_MLP_HEADER = 3'd3;
localparam UART_SEND_MLP_RESULT = 3'd4;
localparam UART_SEND_CONFIDENCE = 3'd5;
localparam UART_DONE = 3'd6;
// 跨时钟域同步：检测mlp_done的上升沿（135MHz -> 27MHz）
// MLP信号跨时钟域同步到27MHz
reg mlp_done_sync_27M;
reg mlp_done_sync_27M_delay;
reg [1:0] mlp_prediction_sync_27M;
reg [31:0] mlp_confidence_sync_27M;

// 同步mlp_done信号并检测上升沿
always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        mlp_done_sync_27M <= 0;
        mlp_done_sync_27M_delay <= 0;
        mlp_prediction_sync_27M <= 0;
        mlp_confidence_sync_27M <= 0;
    end else begin
        // 第一级同步
        mlp_done_sync_27M <= mlp_done;
        mlp_prediction_sync_27M <= mlp_prediction;
        mlp_confidence_sync_27M <= mlp_confidence;
        
        // 第二级同步和延迟用于边沿检测
        mlp_done_sync_27M_delay <= mlp_done_sync_27M;
    end
end

// 检测同步后的mlp_done上升沿
wire mlp_done_rising_sync = mlp_done_sync_27M && !mlp_done_sync_27M_delay;

// 将数值转换为十六进制ASCII字符
function [7:0] byte_to_hex_ascii;
    input [7:0] data;
    input integer nibble; // 0: 高4位, 1: 低4位
    reg [3:0] nibble_data;
    begin
        nibble_data = (nibble == 0) ? data[7:4] : data[3:0];
        if (nibble_data <= 9)
            byte_to_hex_ascii = 8'h30 + nibble_data; // '0'-'9'
        else
            byte_to_hex_ascii = 8'h41 + (nibble_data - 10); // 'A'-'F'
    end
endfunction

// 将16位数值转换为十六进制ASCII字符
function [7:0] word_to_hex_ascii;
    input [15:0] data;
    input integer nibble; // 0: 最高4位, 1: 次高4位, 2: 次低4位, 3: 最低4位
    reg [3:0] nibble_data;
    begin
        case(nibble)
            0: nibble_data = data[15:12];
            1: nibble_data = data[11:8];
            2: nibble_data = data[7:4];
            3: nibble_data = data[3:0];
            default: nibble_data = 4'b0;
        endcase
        if (nibble_data <= 9)
            word_to_hex_ascii = 8'h30 + nibble_data; // '0'-'9'
        else
            word_to_hex_ascii = 8'h41 + (nibble_data - 10); // 'A'-'F'
    end
endfunction

// 将置信度转换为ASCII字符串
function [7:0] confidence_to_ascii;
    input [31:0] conf_q10_22;
    integer conf_int;
    begin
        // 将Q10.22转换为0-100的整数百分比
        conf_int = (conf_q10_22 * 100) >>> Q_FRAC_BITS;
        if (conf_int > 99) conf_int = 99;
        if (conf_int < 0) conf_int = 0;
        
        // 转换为两位ASCII数字
        if (conf_int >= 10) begin
            confidence_to_ascii = 8'h30 + (conf_int / 10);  // 十位
        end else begin
            confidence_to_ascii = 8'h30;  // 十位为0
        end
    end
endfunction

// 获取类型字符串
function [7:0] get_type_char;
    input [1:0] pred;
    input integer pos;
    begin
        case(pred)
            2'b00: get_type_char = (pos == 0) ? "S" : (pos == 1) ? "I" : (pos == 2) ? "N" : " ";  // "SIN"
            2'b01: get_type_char = (pos == 0) ? "S" : (pos == 1) ? "Q" : (pos == 2) ? "U" : " ";  // "SQU"
            2'b10: get_type_char = (pos == 0) ? "T" : (pos == 1) ? "R" : (pos == 2) ? "I" : " ";  // "TRI"
            2'b11: get_type_char = (pos == 0) ? "O" : (pos == 1) ? "T" : (pos == 2) ? "H" : " ";  // "OTH"
            default: get_type_char = " ";
        endcase
    end
endfunction
integer conf_int;
// UART发送状态机 - 合并后的单个always块
always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        uart_state <= UART_IDLE;
        uart_cnt <= 0;
        uart_data <= 0;
        uart_start_send <= 0;
        uart_sending <= 0;
    end else begin
        uart_start_send <= 0;
        
        // 数据选择逻辑 - 根据当前状态和计数器选择要发送的数据
        case(uart_state)
            UART_SEND_ADC_HEADER: begin
                case(uart_cnt)
                    0: uart_data <= "A";
                    1: uart_data <= "D";
                    2: uart_data <= "C";
                    3: uart_data <= ":";
                    default: uart_data <= 0;
                endcase
            end
            
            UART_SEND_ADC_DATA: begin
                // 每个16位数据发送4个十六进制字符
                case(uart_cnt[1:0])
                    0: uart_data <= word_to_hex_ascii(ad_buffer[uart_cnt[9:2]], 0); // 最高4位
                    1: uart_data <= word_to_hex_ascii(ad_buffer[uart_cnt[9:2]], 1); // 次高4位
                    2: uart_data <= word_to_hex_ascii(ad_buffer[uart_cnt[9:2]], 2); // 次低4位
                    3: uart_data <= word_to_hex_ascii(ad_buffer[uart_cnt[9:2]], 3); // 最低4位
                    default: uart_data <= 0;
                endcase
            end
            
            UART_SEND_MLP_HEADER: begin
                case(uart_cnt)
                    0: uart_data <= " ";
                    1: uart_data <= "M";
                    2: uart_data <= "L";
                    3: uart_data <= "P";
                    4: uart_data <= ":";
                    default: uart_data <= 0;
                endcase
            end
            
            UART_SEND_MLP_RESULT: begin
                uart_data <= get_type_char(mlp_prediction_sync_27M, uart_cnt);
            end
            
            UART_SEND_CONFIDENCE: begin
                case(uart_cnt)
                    0: uart_data <= " ";
                    1: uart_data <= confidence_to_ascii(mlp_confidence_sync_27M);  // 十位
                    2: begin
                        
                        conf_int = (mlp_confidence_sync_27M * 100) >>> Q_FRAC_BITS;
                        if (conf_int > 99) conf_int = 99;
                        if (conf_int < 0) conf_int = 0;
                        uart_data <= 8'h30 + (conf_int % 10);  // 个位
                    end
                    3: uart_data <= "%";  // 百分比符号
                    4: uart_data <= "\r"; // 回车
                    5: uart_data <= "\n"; // 换行
                    default: uart_data <= 0;
                endcase
            end
            
            default: uart_data <= 0;
        endcase
        
        // 状态转移控制逻辑
        if (uart_sending && uart_start_send) begin
            uart_start_send <= 0;  // 清除启动信号
        end
        else if (uart_sending && tx_busy_falling) begin
            // 当前字符发送完成，准备发送下一个
            case(uart_state)
                UART_SEND_ADC_HEADER: begin
                    if (uart_cnt < 3) begin
                        uart_cnt <= uart_cnt + 1;
                        uart_start_send <= 1;
                    end else begin
                        uart_state <= UART_SEND_ADC_DATA;
                        uart_cnt <= 0;
                        uart_start_send <= 1;  // 立即开始发送ADC数据
                    end
                end
                
                UART_SEND_ADC_DATA: begin
                    if (uart_cnt < 255) begin  // 64个16位数据 × 4字符 = 256
                        uart_cnt <= uart_cnt + 1;
                        uart_start_send <= 1;
                    end else begin
                        uart_state <= UART_SEND_MLP_HEADER;
                        uart_cnt <= 0;
                        uart_start_send <= 1;  // 立即开始发送MLP头部
                    end
                end
                
                UART_SEND_MLP_HEADER: begin
                    if (uart_cnt < 4) begin
                        uart_cnt <= uart_cnt + 1;
                        uart_start_send <= 1;
                    end else begin
                        uart_state <= UART_SEND_MLP_RESULT;
                        uart_cnt <= 0;
                        uart_start_send <= 1;  // 立即开始发送MLP结果
                    end
                end
                
                UART_SEND_MLP_RESULT: begin
                    if (uart_cnt < 2) begin  // 发送3个字符：0,1,2
                        uart_cnt <= uart_cnt + 1;
                        uart_start_send <= 1;
                    end else begin
                        uart_state <= UART_SEND_CONFIDENCE;
                        uart_cnt <= 0;
                        uart_start_send <= 1;  // 立即开始发送置信度
                    end
                end
                
                UART_SEND_CONFIDENCE: begin
                    if (uart_cnt < 5) begin  // 发送6个字符：0-5
                        uart_cnt <= uart_cnt + 1;
                        uart_start_send <= 1;
                    end else begin
                        uart_state <= UART_DONE;
                        uart_sending <= 0;  // 发送完成
                    end
                end
                
                default: begin
                    uart_state <= UART_IDLE;
                    uart_sending <= 0;
                end
            endcase
        end
        else if (!uart_sending && mlp_done_rising_sync) begin
            // 使用同步后的MLP完成信号启动UART发送
            uart_state <= UART_SEND_ADC_HEADER;
            uart_sending <= 1;
            uart_cnt <= 0;
            uart_start_send <= 1;  // 立即开始发送
            
            // 设置第一个要发送的数据
            uart_data <= "A";
        end
    end
end
// 检测tx_busy下降沿
reg tx_busy_reg;
always @(posedge clk_27M or negedge rst_n) begin
    if (!rst_n) begin
        tx_busy_reg <= 0;
    end else begin
        tx_busy_reg <= tx_busy;
    end
end
wire tx_busy_falling = (!tx_busy) && tx_busy_reg;

// UART发送模块实例化
wire tx_busy;
uart_tx_only my_uart_tx (
    .clk        (clk_27M),
    .temp_data  (uart_data),
    .start_send (uart_start_send),
    .uart_tx    (uart_tx),
    .tx_busy    (tx_busy)
);
endmodule