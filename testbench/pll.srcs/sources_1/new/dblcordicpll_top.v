module dblcordicpll_top(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] adc_data,
    input  wire        adc_valid,
    output wire  [15:0] phase_error,
    output wire        pll_locked,
     // 新增内部信号监测
    output wire signed [15:0] o_pm_sin,
    output wire signed [15:0] o_pm_cos,
    output wire signed [15:0] o_fil_sin, 
    output wire signed [15:0] o_fil_cos,
    output wire [31:0] o_phase,

    output  wire signed [15:0] recon_cos,  // 只取 cos 即可代表重建波形
    output wire signed [15:0] recon_sin,
    output reg               o_recon_vld,  // 重建有效，与 adc_valid 对齐
    output reg signed [15:0] o_err_show,    // 实时误差 = AD – 重建，供波形观察
    
    output wire         [7:0] sin_out,
    output  wire              sin_vld
);

    // 参数定义
    localparam IW = 16;
    localparam PW = 32;  
    localparam OW = 16;
    
    // 内部信号
    reg  [PW-1:0] phase_step;
    reg  load_freq;
    wire signed [IW-1:0] signed_adc;
    reg  [4:0] lg_coeff = 5'd10;
    
    // 状态机
    reg [2:0] state;
    localparam S_IDLE   = 3'd0;
    localparam S_LOAD   = 3'd1;
    localparam S_RUN    = 3'd2;
    
    // 修正的ADC无符号转有符号
    assign signed_adc = adc_data - 16'sd32768;
    
    // 频率控制字 - 对应10kHz
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_step <= 32'd286331;  // 10kHz: (10000/1500000)*2^32
            state <= S_IDLE;
            load_freq <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    state <= S_LOAD;
                    load_freq <= 1'b1;
                end
                S_LOAD: begin
                    load_freq <= 1'b0;
                    state <= S_RUN;
                end
                S_RUN: begin
                    load_freq <= 1'b0;
                    // 保持运行状态
                end
                default: state <= S_IDLE;
            endcase
        end
    end
    
    // PLL锁定检测 - 改进算法
    reg [19:0] locked_counter = 0;
    reg locked = 0;
    reg signed [15:0] last_error = 0;
    wire signed [15:0] diff_phase;
    assign diff_phase = phase_error - last_error;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            locked_counter <= 0;
            locked <= 0;
            last_error <= 0;
        end else if (adc_valid) begin
            // 检查误差变化率
            if (diff_phase < 64 && 
                diff_phase > -64) begin
                if (locked_counter < 20'hFFFFF) 
                    locked_counter <= locked_counter + 1;
            end else begin
                locked_counter <= 0;
            end
            
            last_error <= phase_error;
            
            // 锁定判断 - 需要足够长的稳定时间
            locked <= (locked_counter > 20'd50000); // 约33ms稳定时间
        end
    end
    
    assign pll_locked = locked;
    
    // 双CORDIC PLL实例化 - 添加复位信号
    dblcordicpll #(
        .IW(IW),
        .PW(PW),
        .OW(OW),
        .OPT_TRACK_FREQUENCY(1'b1),
        .OPT_FILTER(1'b1)
    ) pll_inst (
        .i_clk(clk),
        .i_reset(!rst_n),  // 添加复位信号
        .i_ld(load_freq),
        .i_step(phase_step),
        .i_ce(adc_valid),
        .i_input(signed_adc),
        .i_lgcoeff(lg_coeff),
        .o_err(phase_error),
        .o_phase(o_phase)
    );
    ///my
    // 常数幅值 = 满量程 32767
    localparam GAIN = 16'd32767;
    
    wire        recon_busy, recon_done;
    // 第三个 CORDIC：旋转 (GAIN,0) 角度=当前锁定相位
    seqcordic  u_recon (
        .i_clk   (clk),
        .i_reset (~rst_n),
        .i_stb   (adc_valid),      // 与 AD 严格同拍
        .i_xval  (16'd0),
        .i_yval  (GAIN),
        .i_phase (o_phase),        // 32 位锁定相位
        .o_busy  (recon_busy),
        .o_done  (recon_done),
        .o_xval  (recon_cos),      // cosθ
        .o_yval  (recon_sin)                 // sinθ 可留空
    );
    reg signed [15:0] ad_r;
    always @(posedge clk) begin
        if (adc_valid) begin
            ad_r      <= $signed(adc_data) - 16'sd32768; // 转成有符号
            o_recon_vld <= 1'b1;
        end else begin
            o_recon_vld <= 1'b0;
        end
    end
    always @(posedge clk) begin
        if (adc_valid)
            o_err_show <= ad_r - recon_cos;   // 差值，锁定后 ≈ 0
    end
wire signed [15:0] sin_wave;
wire [15:0] sin_tem = (sin_wave + 16'sd32768);
assign sin_out = sin_tem[15:8];
cordic_sin u_sin_gen(
    .i_clk    (clk),
    .i_rst    (~rst_n),
    .i_stb    (adc_valid),     // 与 AD 同拍，也可自己给
    .i_phase   (o_phase),  // 10 kHz @ fs=1.5 MHz
    //.i_phase   (32'd28),  // 10 kHz @ fs=1.5 MHz;=f * 2863.311531
    .o_sin    (sin_wave),
    .o_done   (sin_vld)
);
endmodule