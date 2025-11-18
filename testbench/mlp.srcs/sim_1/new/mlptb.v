`timescale 1ns / 1ps

module tb_top();

// 时钟和复位
reg clk_27M;
reg clk_135M;
reg rst_n;

// ADC输入
reg [7:0] ad_data_in;

// ROM数据输入
reg [31:0] rom_data_in;

// 输出信号
wire ad_clk;
wire uart_tx;
wire [10:0] rom_addr_out;
wire [1:0] mlp_prediction_out;
wire [31:0] mlp_confidence_out;
wire mlp_done_out;
 
// 测试参数
localparam CLK_27M_PERIOD = 37.037;  // 27MHz周期
localparam CLK_135M_PERIOD = 7.407;  // 135MHz周期

// 实例化被测模块
top uut (
    .clk_27M(clk_27M),
    .clk_135M(clk_135M),
    .ad_data_in(ad_data_in),
    .rst_n(rst_n),
    .rom_data_in(rom_data_in),
    .ad_clk(ad_clk),
    .uart_tx(uart_tx),
    .rom_addr(rom_addr_out),
    .mlp_prediction_out(mlp_prediction_out),
    .mlp_confidence_out(mlp_confidence_out),
    .mlp_done_out(mlp_done_out)
);

// 时钟生成
initial begin
    clk_27M = 0;
    forever #(CLK_27M_PERIOD/2) clk_27M = ~clk_27M;
end

initial begin
    clk_135M = 0;
    forever #(CLK_135M_PERIOD/2) clk_135M = ~clk_135M;
end

// ROM数据存储器
reg [31:0] rom_memory [0:2047];

// 读取ROM数据文件（仅第一层权重）
initial begin
    $readmemh("D:/vivadodata/LMLP2/mlp_weights_q10_22_hex.dat", rom_memory);
end

// 方案1：直接单周期延迟
always @(posedge clk_135M) begin
    if (rst_n) begin
        rom_data_in <= rom_memory[rom_addr_out];  // 直接使用当前地址
        //$display("Time: %0t, ROM读取: 地址=%d, 数据=%h", 
                 //$time, rom_addr_out, rom_memory[rom_addr_out]);
    end else begin
        rom_data_in <= 0;
    end
end
    integer i;
    reg [3:0] last_mlp_state;
    reg [31:0] state_timeout_counter;
    reg [3:0] current_state;
    real sin_value;
// 测试激励
initial begin

    
    // 初始化
    rst_n = 0;
    ad_data_in = 8'h00;
    last_mlp_state = uut.MLP_IDLE;
    state_timeout_counter = 0;
    current_state = uut.MLP_IDLE;
    
    // 复位
    #100;
    rst_n = 1;

    // 主测试循环
    repeat (1) begin  // 测试3种波形
        // 生成测试波形数据
        // 生成测试波形数据
        for (i = 0; i < 64; i = i + 1) begin
            @(posedge uut.ad_clk);
            /*
            // 生成 1.5 周期的正弦波
            // 计算正弦波的值
            sin_value = $sin(2 * 3.141592653589793 * 1.2 * i / 64);
            // 将正弦波值映射到 78 到 178 的范围
            ad_data_in = 128 + 30 * sin_value;
            // 确保数据在 78 到 178 之间
            if (ad_data_in < 78) ad_data_in = 78;
            if (ad_data_in > 178) ad_data_in = 178;
            
            if(i < 32)  ad_data_in = 178;
            else  ad_data_in = 78;
*/
            ad_data_in = (i < 32) ? (112 + i):(176 - i);
        end
        
        // 等待MLP处理完成
        while (mlp_done_out != 1) begin
            @(posedge clk_135M);
            
            // 状态监控
            if (uut.mlp_state != last_mlp_state) begin
                last_mlp_state = uut.mlp_state;
                $display("Time: %0t, MLP State: %0d, Neuron: %0d, Input: %0d", 
                         $time, uut.mlp_state, uut.mlp_neuron_idx, uut.mlp_input_idx);
            end
            
            // 超时检测
            if (uut.mlp_state != current_state) begin
                current_state = uut.mlp_state;
                state_timeout_counter = 0;
            end else if (uut.mlp_state != uut.MLP_IDLE && uut.mlp_state != uut.MLP_DONE) begin
                state_timeout_counter = state_timeout_counter + 1;
                
                if (state_timeout_counter > 50000) begin
                    $display("ERROR: Timeout in state %0d at time %0t", uut.mlp_state, $time);
   
                end
            end
        end
        
        // 显示结果
        $display("Time: %0t, MLP Prediction: %0d, Confidence: %0h", 
                 $time, mlp_prediction_out, mlp_confidence_out);
        
        // 等待一段时间再进行下一次测试
        #1000;
    end
    
    $display("Simulation completed successfully");
    $finish;
end

// 仿真控制
initial begin
    #50000000;  // 50ms
    $display("Simulation timeout");
    $finish;
end


endmodule