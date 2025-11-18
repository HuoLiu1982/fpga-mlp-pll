`timescale 1ns / 1ps

module uart_tx_only #(
    parameter BPS_NUM = 16'd234  // 27MHz, 115200 bps
)(
    input         clk,
    input  [7:0]  temp_data,
    input         start_send,
    
    output        uart_tx,
    output        tx_busy
);

// 直接实例化底层UART发送模块
uart_tx #(
    .BPS_NUM(BPS_NUM)
) u_uart_tx (
    .clk      (clk),
    .tx_data  (temp_data),
    .tx_pluse (start_send),
    .uart_tx  (uart_tx),
    .tx_busy  (tx_busy)
);

endmodule