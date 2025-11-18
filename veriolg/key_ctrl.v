module key_ctrl(
    input  wire clk,
    input  wire rst_n,
    input  wire key_in,        // 按键输入（低电平有效）
    output reg  key_short,     // 短按触发（单周期脉冲）
    output reg  key_long       // 长按触发（单周期脉冲）
   );

    parameter CLK_FREQ = 27_000_000;
    parameter DEBOUNCE_MS = 20;
    parameter HOLD_MS = 1000;

    parameter CNT_20MS = (CLK_FREQ/1000)*DEBOUNCE_MS;   // 20ms去抖
    parameter CNT_1S   = (CLK_FREQ/1000)*HOLD_MS;       // 1秒长按判定

    reg key_sync0, key_sync1;
    reg key_state;          // 当前稳定按键状态
    reg [31:0] cnt_debounce;
    reg [31:0] cnt_hold;

    // ===================== 信号同步（防亚稳态） =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            key_sync0 <= 1'b1;
            key_sync1 <= 1'b1;
        end else begin
            key_sync0 <= key_in;
            key_sync1 <= key_sync0;
        end
    end

    // ===================== 20ms去抖动 =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_debounce <= 0;
            key_state <= 1'b1; // 初始未按下
        end else begin
            if (key_sync1 != key_state) begin
                cnt_debounce <= cnt_debounce + 1;
                if (cnt_debounce >= CNT_20MS) begin
                    key_state <= key_sync1; // 状态确认
                    cnt_debounce <= 0;
                end
            end else begin
                cnt_debounce <= 0;
            end
        end
    end

    // ===================== 长按 / 短按检测 =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_hold  <= 32'd0;
            key_short <= 1'b0;
            key_long  <= 1'b0;
        end else begin
            // 默认清单周期脉冲
            key_short <= 1'b0;
            key_long  <= 1'b0;

            if (key_state == 1'b0) begin
                // 按键稳定按下期间
                if (cnt_hold < CNT_1S)
                    cnt_hold <= cnt_hold + 1;
                else
                    cnt_hold <= CNT_1S; // 饱和

                // 到达阈值时产生长按脉冲（仅一次）
                if (cnt_hold == CNT_1S - 1)
                    key_long <= 1'b1;
            end else begin
                // 松开时，如果按下时间在阈值以内，则产生短按脉冲
                if ((cnt_hold > 0) && (cnt_hold < CNT_1S))
                    key_short <= 1'b1;

                // 松开后清计数器
                cnt_hold <= 32'd0;
            end
        end
    end

endmodule