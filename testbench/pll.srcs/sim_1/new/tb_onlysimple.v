`timescale 1ns / 1ps

module tb_simple_test;

    // 基本参数
    localparam CLK_PERIOD = 7.407;  // 135MHz
    
    // 信号定义
    reg clk;
    reg rst_n;
    reg i_ce;
    reg signed [15:0] i_input;
    wire [15:0] o_err;
    wire signed [15:0] o_pm_sin, o_pm_cos;
    
    // 直接实例化dblcordicpll进行测试
    dblcordicpll #(
        .IW(16),
        .PW(32),
        .OW(16),
        .OPT_TRACK_FREQUENCY(1'b1),
        .OPT_FILTER(1'b1)
    ) dut (
        .i_clk(clk),
        .i_ld(1'b0),
        .i_step(32'd28633115),
        .i_ce(i_ce),
        .i_input(i_input),
        .i_lgcoeff(5'd4),
        .o_err(o_err),
        .o_pm_sin(o_pm_sin),
        .o_pm_cos(o_pm_cos)
    );
    
    // 时钟生成
    always #(CLK_PERIOD/2) clk = ~clk;
        integer i;    
    // 测试序列
    initial begin

        
        // 初始化
        clk = 0;
        rst_n = 0;
        i_ce = 0;
        i_input = 0;
        
        // 生成VCD
        $dumpfile("simple_test.vcd");
        $dumpvars(0, tb_simple_test);
        
        // 复位
        #100;
        rst_n = 1;
        #100;
        
        $display("简单测试开始");
        $display("时间(ns)\ti_input\to_pm_sin\to_pm_cos\to_err");
        
        // 发送简单的直流信号测试
        for (i = 0; i < 1000; i = i + 1) begin
            @(posedge clk);
            i_ce = (i % 90 == 0);  // 1.5MHz采样率
            
            if (i_ce) begin
                i_input = 1000;  // 简单直流信号
                
                $display("%0t\t%h\t%h\t%h\t%h", 
                    $time, i_input, o_pm_sin, o_pm_cos, o_err);
            end
        end
        
        $display("简单测试完成");
        $finish;
    end

endmodule