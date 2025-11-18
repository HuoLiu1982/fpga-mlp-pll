`timescale 1ns/1ps

module dblcordicpll_top (
    // 时钟和复位
    input  wire         clk,           // 系统时钟
    input  wire         reset_n,       // 异步复位，低电平有效
    input  wire         enable_posedge,        // AD采样时钟的上升沿（数据有效信号）
    
    // 输入信号
    input  wire signed [15:0] signal_in,  // 输入信号
    input  wire          r_error_jump,        // 误差突变标志 
    // 输出信号
    output wire [15:0]  phase_error,   // 相位误差，稳定值2**15
    output wire [31:0]  output_phase,   // 输出相位
    output wire         pll_done
    
);

    // 参数定义
    parameter IW = 16;      // 输入位宽
    parameter PW = 32;      // 相位位宽  
    parameter OW = 16;      // 输出位宽
    
    // 增益系数定义 - 从3到14
    localparam GAIN_3  = 5'd3;     // 增益3
    localparam GAIN_4  = 5'd4;     // 增益4
    localparam GAIN_5  = 5'd5;     // 增益5
    localparam GAIN_6  = 5'd6;     // 增益6
    localparam GAIN_7  = 5'd7;     // 增益7
    localparam GAIN_8  = 5'd8;     // 增益8
    localparam GAIN_9  = 5'd9;     // 增益9
    localparam GAIN_10 = 5'd10;    // 增益10
    localparam GAIN_11 = 5'd11;    // 增益11
    localparam GAIN_12 = 5'd12;    // 增益12
    localparam GAIN_13 = 5'd13;    // 增益13
    localparam GAIN_14 = 5'd14;    // 增益14
    
    // 持续时间倍数定义 - 按照您的要求调整
    localparam DUL_3_TO_4   = 32'd3;    // 3->4: 1倍
    localparam DUL_4_TO_5   = 32'd3;    // 4->5: 1倍
    localparam DUL_5_TO_6   = 32'd3;    // 5->6: 1倍
    localparam DUL_6_TO_7   = 32'd3;    // 6->7: 3倍
    localparam DUL_7_TO_8   = 32'd3;    // 7->8: 3倍
    localparam DUL_8_TO_9   = 32'd3;    // 8->9: 3倍
    localparam DUL_9_TO_10  = 32'd12;   // 9->10: 12倍
    localparam DUL_10_TO_11 = 32'd9;    // 10->11: 9倍
    localparam DUL_11_TO_12 = 32'd9;    // 11->12: 9倍
    localparam DUL_12_TO_13 = 32'd15;   // 12->13: 15倍
    localparam DUL_13_TO_14 = 32'd15;   // 13->14: 15倍
    localparam DUL_14_FINAL = 32'd15;   // 14保持: 15倍
    
    // 基础持续时间
    localparam BASE_DUL_LEN = 32'd10_000; // 基础持续时间10ms
    localparam INIT_CNT = 16'd3000;    // 初始不计入计数
    
    // 误差突变检测阈值
    localparam ERROR_THRESHOLD = 16'd100; // 误差突变阈值（接近2**15=32768）
    
    // 内部信号声明
    reg                 pll_id;              // PLL加载信号
    reg [PW-1:0]        r_phase_step;        // 相位步进寄存器
    reg [4:0]           r_lg_coeff;          // 增益系数寄存器
    
    // 增益控制状态机
    reg [3:0]           r_stage;             // 增益控制阶段（0-11共12个阶段）
    reg [15:0]          r_ice_cnt;           // 初始计数器
    reg [31:0]          r_dul_cnt;           // 持续时间计数器
    reg [31:0]          r_current_dul_len;   // 当前阶段持续时间
    reg [15:0]          r_prev_error_abs;    // 上一周期误差绝对值
    
    
    
    // 计算当前误差绝对值
    reg [15:0] current_error_abs;
    // 固定相位步进值
    localparam FIXED_PHASE_STEP = 32'd429496;
    
    
    // 实例化PLL核心模块
    dblcordicpll #(
        .IW(IW),
        .PW(PW), 
        .OW(OW),
        .OPT_TRACK_FREQUENCY(1'b1),
        .OPT_FILTER(1'b1)
    ) u_pll_core (
        .i_clk        (clk),
        .i_reset      (~reset_n),        //高复位
        .i_ld         (pll_id),     // 应该在复位之后，脉冲
        .i_step       (r_phase_step),
        .i_ce         (enable_posedge), // 使用ad上升沿作为PLL使能
        .i_input      (signal_in),   // 使用同步后的输入信号
        .i_lgcoeff    (r_lg_coeff),
        .o_err        (phase_error),
        .o_phase      (output_phase),
        .pd_done      (pll_done) 
    );
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n)begin
            pll_id <= 1;
        end else begin
            if (pll_id == 1) begin
                pll_id <= 0;
            end else if(r_error_jump)begin
               pll_id <= 1;
            end
        end
    end
    // 误差突变检测逻辑
    /*
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_prev_error_abs <= 16'd0;
            r_error_jump <= 1'b0;
        end else if (enable_posedge) begin  // 只在ad上升沿时检测
            current_error_abs = phase_error;           
            // 检测误差突变：只在最终阶段检测
            if (r_stage == 4'd11 && (current_error_abs > phase_error + ERROR_THRESHOLD
                                     || current_error_abs < phase_error - ERROR_THRESHOLD)) begin
                r_error_jump <= 1'b0;
            end else begin
                r_error_jump <= 1'b0;
            end
            
            r_prev_error_abs <= current_error_abs;
        end
    end
    */
    // 增益控制状态机 - 从增益3到14共12个阶段
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_stage          <= 4'd0;
            r_ice_cnt        <= 16'd0;
            r_dul_cnt        <= 32'd0;
            r_lg_coeff       <= GAIN_3;
            r_current_dul_len <= BASE_DUL_LEN * DUL_3_TO_4;
        end else if (r_error_jump) begin
            // 误差突变：重置到第一阶段
            r_stage          <= 4'd0;
            r_ice_cnt        <= 16'd0;
            r_dul_cnt        <= 32'd0;
            r_lg_coeff       <= GAIN_3;
            r_current_dul_len <= BASE_DUL_LEN * DUL_3_TO_4;
        end else if (enable_posedge) begin  // 只在enable上升沿时更新状态机
            // 初始计数阶段
            if (r_ice_cnt < INIT_CNT) begin
                r_ice_cnt <= r_ice_cnt + 16'd1;
                r_dul_cnt <= 32'd0;
            end else begin
                r_ice_cnt <= INIT_CNT + 16'd1; // 保持固定值
                r_dul_cnt <= r_dul_cnt + 32'd1;
                
                // 状态转移逻辑 - 从增益3到14共12个阶段
                case (r_stage)
                    4'd0: begin // 阶段0：增益3
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd1;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_4;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_4_TO_5;
                        end
                    end
                    
                    4'd1: begin // 阶段1：增益4
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd2;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_5;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_5_TO_6;
                        end
                    end
                    
                    4'd2: begin // 阶段2：增益5
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd3;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_6;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_6_TO_7;
                        end
                    end
                    
                    4'd3: begin // 阶段3：增益6
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd4;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_7;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_7_TO_8;
                        end
                    end
                    
                    4'd4: begin // 阶段4：增益7
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd5;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_8;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_8_TO_9;
                        end
                    end
                    
                    4'd5: begin // 阶段5：增益8
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd6;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_9;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_9_TO_10;
                        end
                    end
                    
                    4'd6: begin // 阶段6：增益9
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd7;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_10;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_10_TO_11;
                        end
                    end
                    
                    4'd7: begin // 阶段7：增益10
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd8;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_11;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_11_TO_12;
                        end
                    end
                    
                    4'd8: begin // 阶段8：增益11
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd9;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_12;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_12_TO_13;
                        end
                    end
                    
                    4'd9: begin // 阶段9：增益12
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd10;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_13;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_13_TO_14;
                        end
                    end
                    
                    4'd10: begin // 阶段10：增益13
                        if (r_dul_cnt >= r_current_dul_len) begin
                            r_stage          <= 4'd11;
                            r_dul_cnt        <= 32'd0;
                            r_lg_coeff       <= GAIN_14;
                            r_current_dul_len <= BASE_DUL_LEN * DUL_14_FINAL;
                        end
                    end
                    
                    4'd11: begin // 阶段11：增益14（最终阶段）
                        // 保持当前增益，不再切换
                        // 等待误差突变强制复位
                    end
                    
                    default: begin
                        r_stage          <= 4'd0;
                        r_lg_coeff       <= GAIN_3;
                        r_current_dul_len <= BASE_DUL_LEN * DUL_3_TO_4;
                    end
                endcase
            end
        end
    end
    
    // 相位步进配置（固定值）
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            r_phase_step <= FIXED_PHASE_STEP;
        end else begin
            r_phase_step <= FIXED_PHASE_STEP; // 保持固定
        end
    end

endmodule