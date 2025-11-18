`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/10/13 18:35:24
// Design Name: 
// Module Name: cordic_sin
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


`default_nettype none

module cordic_sin #(
    parameter OW = 16,            // 输出位宽
    parameter PW = 32             // 相位位宽
)(
    input  wire             i_clk,
    input  wire             i_rst,      // 高复位
    input  wire             i_stb,      // 来一单脉冲
    input  wire [PW-1:0]    i_phase,     // 频率字，单位：2^-32 * fs
    output reg  signed [OW-1:0] o_sin,  // 正弦输出
    output reg              o_done      // 单脉冲完成标志
);

// ---------- 相位累加器 ----------
reg [PW-1:0] phase;
always @(posedge i_clk) begin
    if (i_rst)      phase <= 0;
    else if (i_stb) phase <=  i_phase;   // 含频率修正
end

// ---------- CORDIC 旋转 (1,0) ----------
localparam GAIN = 16'd32767;          // 满幅值
wire signed [OW-1:0] cos_out, sin_out;
wire busy, done;

seqcordic  u_cordic (
    .i_clk   (i_clk),
    .i_reset (i_rst),
    .i_stb   (i_stb),
    .i_xval  (16'd0),        // (GAIN, 0)
    .i_yval  (GAIN),
    .i_phase (phase),       // 当前总相位
    .o_busy  (busy),
    .o_done  (done),
    .o_xval  (cos_out),     // cos(phase)
    .o_yval  (sin_out)      // sin(phase) ← 我们要的
);

// ---------- 输出寄存 ----------
always @(posedge i_clk) begin
    if (done) begin
        o_sin  <= sin_out;
        o_done <= 1'b1;
    end else
        o_done <= 1'b0;
end

endmodule
