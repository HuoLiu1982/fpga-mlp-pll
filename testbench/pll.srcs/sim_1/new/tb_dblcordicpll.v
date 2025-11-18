`timescale 1ns / 1ps

module tb_dblcordicpll;

    // 时钟参数
    localparam CLK_PERIOD = 7.407;  // 135MHz
    localparam SAMPLE_PERIOD = 666.666;  // 1.5MHz
    
    // 测试参数
    localparam TEST_FREQ = 1000;  // 10kHz
    localparam NUM_SAMPLES = 300000;//3000
    
    // 信号定义
    reg clk;
    reg rst_n;
    reg [15:0] adc_data;
    reg adc_valid;
    wire  [15:0] phase_error;
    wire pll_locked;
    
    // 改进的测试变量
    integer i;
    real phase;
    real sin_value;
    integer sample_count = 0;
    
    // 时钟生成
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // 新增内部信号监测
    wire signed [15:0] pm_sin, pm_cos;
    wire signed [15:0] fil_sin, fil_cos;
    wire [31:0] phase_accum;
    wire signed [15:0] o_recon;     // 重建样本，与 adc_data 同拍
    wire               o_recon_vld;  // 重建有效，与 adc_valid 对齐
    wire signed [15:0] o_err_show;    // 实时误差 = AD - 重建，供波形观察
    // 实例化顶层模块 - 连接监测信号
    dblcordicpll_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .adc_data(adc_data),
        .adc_valid(adc_valid),
        .phase_error(phase_error),
        .pll_locked(pll_locked),
        
        // 连接监测信号
        .o_pm_sin(pm_sin),
        .o_pm_cos(pm_cos),
        .o_fil_sin(fil_sin),
        .o_fil_cos(fil_cos),
        .o_phase(phase_accum),

        .o_recon_vld(o_recon_vld),
        .o_err_show(o_err_show)
    );
    
    // 采样时钟生成
    reg [9:0] sample_counter = 0;
    always @(posedge clk) begin
        if (!rst_n) begin
            sample_counter <= 0;
            adc_valid <= 0;
        end else begin
            if (sample_counter == 89) begin // 135MHz/1.5MHz = 90
                sample_counter <= 0;
                adc_valid <= 1;
            end else begin
                sample_counter <= sample_counter + 1;
                adc_valid <= 0;
            end
        end
    end
    
    // 测试激励
    initial begin
        // 初始化
        clk = 0;
        rst_n = 0;
        adc_data = 0;
        phase = 0;
        
        // 复位
        #100;
        rst_n = 1;
        
        $display("开始仿真 - 测试频率: %0d Hz", TEST_FREQ);
        $display("时间(ns)\tADC数据\t相位误差\t锁定状态\tPM_SIN\tPM_COS\tFIL_SIN\tFIL_COS\tPHASE_ACCUM");
        
        // 生成测试信号
        for (i = 0;; i = i + 1) begin
            @(posedge adc_valid); // 等待采样时钟
            
            // 计算正弦波 - 改进的算法
            phase = 2.0 * 3.1415926535 * TEST_FREQ * i / 1500000.0;
            sin_value = $sin(phase);
            
            // 生成ADC数据 - 使用正确的幅度
            adc_data = 16'd32768 + $rtoi(sin_value * 32767);
            
            sample_count = sample_count + 1;
            
            // 显示监测数据 - 包含内部信号
            if (sample_count % 100 == 0) begin
                $display("%0t\t%h\t%h\t%b\t%h\t%h\t%h\t%h\t%h", 
                    $time, adc_data, phase_error, pll_locked,
                    pm_sin, pm_cos, fil_sin, fil_cos, phase_accum[31:16]);
            end
            
            // 等待一些时钟周期让CORDIC完成计算
            #(CLK_PERIOD * 50);
        end
        
        $display("仿真完成 - 最终相位误差: %h (%0d)", phase_error, phase_error);
        $display("最终内部信号 - PM_SIN: %h, PM_COS: %h, FIL_SIN: %h, FIL_COS: %h", 
                pm_sin, pm_cos, fil_sin, fil_cos);
        $finish;
    end
    
    // 实时监测信号变化
    always @(posedge adc_valid) begin
        if (pll_locked && ($random % 100 == 0)) begin
            $display("锁定状态监测: 时间=%0t", $time, phase_error, pm_sin);
            $display("仿真完成 - 最终相位误差: %h (%0d)", phase_error, phase_error);
            $display("最终内部信号 - PM_SIN: %h, PM_COS: %h, FIL_SIN: %h, FIL_COS: %h", 
                pm_sin, pm_cos, fil_sin, fil_cos);
                $finish;
        end
    end
    
    // VCD文件
    initial begin
        $dumpfile("dblcordicpll.vcd");
        $dumpvars(0, tb_dblcordicpll);
    end
    
    // 超时保护
    initial begin
        #50000000;  // 50ms
        $display("仿真超时!");
        $display("当前状态: rst_n=%b, adc_valid=%b, phase_error=%h", rst_n, adc_valid, phase_error);
        $display("内部信号: pm_sin=%h, pm_cos=%h, fil_sin=%h, fil_cos=%h", 
                pm_sin, pm_cos, fil_sin, fil_cos);
        $finish;
    end

endmodule