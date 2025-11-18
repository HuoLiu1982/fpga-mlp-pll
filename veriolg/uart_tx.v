`timescale 1ns / 1ps

module uart_tx #(
    parameter BPS_NUM = 16'd234
)(
    input          clk,
    input [7:0]    tx_data,
    input          tx_pluse,
                   
    output reg     uart_tx,
    output         tx_busy
);

//==========================================================================
//wire and reg in the module
//==========================================================================
reg             tx_pluse_reg = 0;
reg [2:0]       tx_bit_cnt = 0;
reg [2:0]       tx_state = 0;
reg [2:0]       tx_state_n = 0;
reg [15:0]      clk_div_cnt = 0;

// uart tx state machine's state
localparam  IDLE        = 3'h0;
localparam  SEND_START  = 3'h1;
localparam  SEND_DATA   = 3'h2;
localparam  SEND_STOP   = 3'h3;
localparam  SEND_END    = 3'h4;

//==========================================================================
//logic
//==========================================================================
assign tx_busy = (tx_state != IDLE);

always @(posedge clk) begin
    tx_pluse_reg <= tx_pluse;
end

// 波特周期计数器
always @(posedge clk) begin
    if (clk_div_cnt == BPS_NUM || (!tx_pluse_reg && tx_pluse))
        clk_div_cnt <= 16'h0;
    else
        clk_div_cnt <= clk_div_cnt + 16'h1;
end

// 发送bit位计数
always @(posedge clk) begin
    if (tx_state == IDLE)
        tx_bit_cnt <= 3'h0;
    else if ((tx_bit_cnt == 3'h7) && (clk_div_cnt == BPS_NUM))
        tx_bit_cnt <= 3'h0;
    else if ((tx_state == SEND_DATA) && (clk_div_cnt == BPS_NUM))
        tx_bit_cnt <= tx_bit_cnt + 3'h1;
    else 
        tx_bit_cnt <= tx_bit_cnt;
end

//==========================================================================
// 状态机
//==========================================================================
always @(posedge clk) begin
    tx_state <= tx_state_n;
end

always @(*) begin
    case(tx_state)
        IDLE: begin
            if (!tx_pluse_reg && tx_pluse)
                tx_state_n = SEND_START;
            else
                tx_state_n = tx_state;
        end
        SEND_START: begin
            if (clk_div_cnt == BPS_NUM)
                tx_state_n = SEND_DATA;
            else
                tx_state_n = tx_state;
        end
        SEND_DATA: begin
            if (tx_bit_cnt == 3'h7 && clk_div_cnt == BPS_NUM)
                tx_state_n = SEND_STOP;
            else
                tx_state_n = tx_state;
        end
        SEND_STOP: begin
            if (clk_div_cnt == BPS_NUM)
                tx_state_n = SEND_END;
            else
                tx_state_n = tx_state;
        end
        SEND_END: tx_state_n = IDLE;
        default: tx_state_n = IDLE;
    endcase
end

// 状态机输出
always @(posedge clk) begin
    if (tx_state != IDLE) begin
        case(tx_state)
            IDLE:       uart_tx <= 1'h1;
            SEND_START: uart_tx <= 1'h0;
            SEND_DATA: begin
                case(tx_bit_cnt)
                    3'h0: uart_tx <= tx_data[0];
                    3'h1: uart_tx <= tx_data[1];
                    3'h2: uart_tx <= tx_data[2];
                    3'h3: uart_tx <= tx_data[3];
                    3'h4: uart_tx <= tx_data[4];
                    3'h5: uart_tx <= tx_data[5];
                    3'h6: uart_tx <= tx_data[6];
                    3'h7: uart_tx <= tx_data[7];
                    default: uart_tx <= 1'h1;
                endcase
            end
            SEND_STOP: uart_tx <= 1'h1;
            default: uart_tx <= 1'h1;
        endcase
    end
    else begin
        uart_tx <= 1'h1;
    end
end

endmodule