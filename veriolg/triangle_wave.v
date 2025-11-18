module triangle_wave(
    input clk,
    input [31:0] phase,
    output reg signed [15:0] triangle_out
);

    // 直接使用相位的高16位作为线性映射
    wire [15:0] linear;
    assign linear = phase[31:16];
    
    // 三角波生成：根据最高位决定斜率
    always @(posedge clk) begin
        if (phase[31] == 1'b0) begin
            // 前半周期：上升
            triangle_out <= {1'b0, linear[14:0]} - 16'd16384; // -16384 to +16383
        end else begin
            // 后半周期：下降  
            triangle_out <= 16'd16384 - {1'b0, linear[14:0]}; // +16383 to -16384
        end
    end

endmodule