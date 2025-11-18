// ==============================================================================================================================
// ADC-DBLCORDICPLL-DDS
//V1.0--基础功能实现，放大10mv到1v（外置放大器）
//      波形跳变方波（高低电平为正弦波）：dac无符号处理缺失
//      针对1k-10k(50k 30s 失败,信号放大到100mvpp依旧不锁定，信号频率接近)
//V2.0--修改信号显示，波形完整无变形（低频略微扭曲）
//      100-100_000均可以锁相（重新复位，低频尝试真正稳定不晃动需要时间约60s，高频很快稳定，修正时间分配）
//      波形跳变依旧不行：经常向增高方向跳变（怀疑跟符号数"-correction"有关，100K到99K）（但是有时向低处锁相，变化迟缓，接近时又向高处锁，查看循环和反馈）
//      方波、三角波成功锁相；不同波形跳变（向高频可以，时间看距离和当前频率，有点像是趋近频率过程中前面跳过了正确频率就不会回头的样子）看情况：给跳变修改逻辑
//V3.0--波形识别不对，在于数据转换，直接使用低7位不行，应该取高7位并且加上偏置64;并且识别波形后不会处理信号
//      跳变依旧很慢:感觉出错在复位（已经尝试了10LSB）  
//V4.0--波形识别不准（ad采集成功，但是噪声很大）；滤波，或者修改训练代码，加入大量噪声版本。
//      波形显示还是不行
//      跳变不在复位，50LSB太小；感觉error不是32768，是0；看仿真 
//      新问题，三角波锁不上
//V5.0--用时间序列的确定性来替代逻辑判断的复杂性：
//      波形显示正确，但是mlp识别不准（一是信号太小，二是频率问题）；
//      虽然跳变正常，但是跳的太快，波形会跳动（如何修改？增加后面的时间，等效于2s一次重置？或者研究根据误差选择是否跳变）                                                                                                              统一减少时间
// ===============================================================================================================================
module top (
    input  wire        clk_27M,     // 27MHz主时钟
    input  wire [15:0] adc_dai_A,   // ADC输入数据
    input  wire        rst_n,       // 复位信号，低有效
	input  wire		   key_change,	//key4,低电平有效
    output wire        da_clk,      // DAC时钟
    output wire [7:0]  da_data_out, // DAC输出数据
    output wire        adc_clk_A,   // ADC时钟
    output wire        uart_tx
);

// 参数定义
parameter IW = 16;
parameter PW = 32;
parameter OW = 16;
parameter LARGE_NUM = 8;
parameter SCALE_FACTOR = 256;  // 2^8
localparam GAIN = 16'd32767;          // 2**14
// 内部信号定义
wire clk_100M;
wire lock_100M;
reg signed [IW-1:0] signal_in;

// =================================================
// PLL模块 - 生成100MHz时钟
// =================================================
pll_100M my_pll_100M (
  .clkout0(clk_100M),    // 输出100MHz时钟
  .lock(lock_100M),      // PLL锁定信号
  .clkin1(clk_27M)       // 输入27MHz时钟
);
// =================================================
// key模块 - 产生脉冲
// =================================================
wire key_short;
reg  key_syn1;
reg  key_reg;
wire key_pulse;//100M上升沿,检测按键上升沿
key_ctrl u_keymode(
	.clk(clk_27M), 
	.rst_n(rst_n), 
	.key_in(key_change), //低电平有效
	.key_short(key_short), //高电平有效
	.key_long()
);
always @(posedge clk_100M or negedge rst_n)begin
	if(!rst_n)begin
		key_reg <= 0;
		key_syn1 <= 0;
	end else begin
		key_syn1 <= key_short;
		key_reg <= key_syn1;
	end
end
assign key_pulse = key_syn1 && (!key_reg);
// =================================================
// AD模块 - 数据采集和处理
// =================================================
reg [7:0] ad_cnt;
reg [15:0] ad_data_in;
reg signed [IW:0] ad_bias;
reg signed [(IW+LARGE_NUM-1):0] amp;

always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        ad_cnt <= 8'd0;
        ad_data_in <= 16'd0;
    end else begin
        if (ad_cnt == 8'd99) begin  // 每100个周期采样一次
            ad_data_in <= adc_dai_A;
            ad_cnt <= 8'd0;
        end else begin
            ad_cnt <= ad_cnt + 8'd1;
        end
    end
end

// 信号处理：无符号转有符号并放大 - 流水线化
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        ad_bias <= 0;
        amp <= 0;
        signal_in <= 0;
    end else begin
        // 第一级：偏置处理
        ad_bias <= ad_data_in - 16'd32768;
        
        // 第二级：放大处理
        amp <= ad_bias * SCALE_FACTOR;
        
        // 第三级：饱和处理
        if (amp > 23'sd32767) begin
            signal_in <= 16'sd32767;
        end else if (amp < -23'sd32768) begin
            signal_in <= -16'sd32768;
        end else begin
            signal_in <= amp[15:0];
        end
    end
end

assign adc_clk_A = clk_100M;

// =================================================
// 时钟分频模块 - 生成1MHz时钟
// =================================================
parameter DIVIDER = 100;
parameter CNT_MAX = DIVIDER/2 - 1;

reg clk_1M;
reg [7:0] clk_cnt;
reg clk_1M_prev;
reg pulse_ad;

always @(posedge clk_100M or negedge rst_n) begin
    if(!rst_n) begin
        clk_cnt <= 8'd0;
        clk_1M <= 1'b0;
        clk_1M_prev <= 1'b0;
        pulse_ad <= 1'b0;
    end else begin
        clk_1M_prev <= clk_1M;
        
        if(clk_cnt == CNT_MAX) begin
            clk_1M <= ~clk_1M;
            clk_cnt <= 8'd0;
        end else begin
            clk_cnt <= clk_cnt + 8'd1;
        end
        
        pulse_ad <= (clk_1M && !clk_1M_prev);
    end
end

// =================================================
// 信号识别模块 - 优化时序逻辑
// =================================================
//ad_bias(200-1000);除以8;加上128
// ad_bias处理流水线
wire [15:0]ad_lmlp = signal_in + 16'd32768;

reg enable_mlp;
reg [1:0] mlp_prediction_sync;
reg mlp_done_sync;
reg mlp_done_prev;
reg [1:0] ad_type;
reg [1:0] mlp_state;

// lmlp例化
wire [1:0] mlp_prediction;
wire mlp_done;
wire mlp_done_rising;
/*
reg ad_change_sync;
reg ad_change_prev;
wire ad_change;
wire ad_change_rising;
*/
localparam MLP_IDLE = 2'd0;
localparam MLP_WAIT_START = 2'd1;
localparam MLP_WAIT_DONE = 2'd2;
/*
reg [7:0] lmlp_ad;
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        ad_lmlp <= 16'd0;
        lmlp_ad <= 8'd0;
    end else begin
        // 第一级：右移3位
        ad_lmlp <= $signed(ad_bias >>> 2) + 8'sd128;
        // 第二级：取8位
        lmlp_ad <= ad_lmlp[7:0];
    end
end
*/
// lmlp例化 - 使用同步后的数据
lmlp_top u_lmlp(
    .clk_27M(clk_27M),
    .clk_135M(clk_100M),
    .ad_data_in(adc_dai_A),  // 使用同步后的数据
    .rst_n(rst_n && ~enable_mlp),
    .mlp_prediction(mlp_prediction),
    .mlp_done(mlp_done),
    .uart_tx(uart_tx)
);

// 跨时钟域同步：MLP输出同步到100MHz时钟域
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        mlp_prediction_sync <= 2'b0;
        mlp_done_sync <= 1'b0;
        mlp_done_prev <= 1'b0;
    end else begin
        mlp_prediction_sync <= mlp_prediction;
        mlp_done_sync <= mlp_done;
        mlp_done_prev <= mlp_done_sync;
    end
end

assign mlp_done_rising = mlp_done_sync && !mlp_done_prev;
/*
// ad_change信号同步和边沿检测
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        ad_change_sync <= 1'b0;
        ad_change_prev <= 1'b0;
    end else begin
        ad_change_sync <= ad_change;
        ad_change_prev <= ad_change_sync;
    end
end

assign ad_change_rising = ad_change_sync && !ad_change_prev;
*/
// enable_mlp状态机 - 改为时序逻辑
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        mlp_state <= MLP_IDLE;
        enable_mlp <= 1'b0;
        ad_type <= 2'b0;
    end else begin
        case (mlp_state)
            MLP_IDLE: begin
                enable_mlp <= 1'b0;               
                if (key_pulse) begin
                    mlp_state <= MLP_WAIT_START;
                    enable_mlp <= 1'b1;  // 启动MLP，单周期
                end
            end
            
            MLP_WAIT_START: begin
                // 等待MLP开始工作，然后进入等待完成状态
                enable_mlp <= 1'b0;
                if (mlp_done_rising) begin
                    mlp_state <= MLP_WAIT_DONE;
                end
            end
            
            MLP_WAIT_DONE: begin
               // MLP完成，更新波形类型
               ad_type <= mlp_prediction_sync;
               mlp_state <= MLP_IDLE;
            end
            
            default: begin
                mlp_state <= MLP_IDLE;
                enable_mlp <= 1'b0;
            end
        endcase
    end
end

// =================================================
// PLL处理模块
// =================================================
wire [OW-1:0] phase_error;
wire [PW-1:0] output_phase;
wire pll_done;

dblcordicpll_top pll_dut(
    .clk(clk_100M),
    .reset_n(rst_n),
    .enable_posedge(pulse_ad),
    .signal_in(signal_in),
	.r_error_jump(key_pulse),        // 误差突变标志    
    .phase_error(phase_error),
    .output_phase(output_phase),
    .pll_done(pll_done)
);

// =================================================
// DDS DAC模块
// =================================================
reg [1:0] pll_to_dds_cnt;
reg dds_enable;
reg signed [OW-1:0] o_output;
reg [31:0] output_phase_d1;
reg signed [OW-1:0] o_output_squ;
reg [7:0] da_data;

wire signed [OW-1:0] o_output_sin;
wire signed [OW-1:0] o_output_tri;
wire [15:0] da_unsigned;
wire [16:0] da_round;

// DDS使能控制状态机
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        pll_to_dds_cnt <= 2'b0;
        dds_enable <= 1'b0;
    end else begin
        case (pll_to_dds_cnt)
            2'd0: begin
                if (pll_done) begin
                    pll_to_dds_cnt <= 2'd1;
                end
                dds_enable <= 1'b0;
            end
            2'd1: begin
                pll_to_dds_cnt <= 2'd2;
                dds_enable <= 1'b0;
            end
            2'd2: begin
                pll_to_dds_cnt <= 2'd0;
                dds_enable <= 1'b1;
            end
            default: begin
                pll_to_dds_cnt <= 2'd0;
                dds_enable <= 1'b0;
            end
        endcase
    end
end

// sin_dds
seqcordic sin_dds (
    .i_clk(clk_100M),
    .i_reset(!rst_n),
    .i_stb(dds_enable),
    .i_xval(GAIN),
    .i_yval(16'd0),
    .i_phase(output_phase),
    .o_xval(o_output_sin)
);

// 方波和三角波生成 - 流水线化
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        output_phase_d1 <= 32'd0;
        o_output_squ <= 16'd0;
    end else begin
        output_phase_d1 <= output_phase; // 相位流水线
        
        // 方波生成
        if (output_phase_d1 < 32'd2147483648) begin
            o_output_squ <= -GAIN;
        end else begin
            o_output_squ <= GAIN;
        end
    end
end

triangle_wave tri_dds (
    .clk(clk_100M),
    .phase(output_phase_d1), // 使用流水线后的相位
    .triangle_out(o_output_tri)
);

// 波形选择
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
       o_output <= 0; 
    end else begin
        case(ad_type)
            2'b00: o_output <= o_output_sin;
            2'b01: o_output <= o_output_squ;
            2'b10: o_output <= o_output_tri;
            default: o_output <= o_output_sin;
        endcase
    end
end

// =================================================
// DAC输出,16位有符号o_output四舍五入变成8位无符号da_data
// =================================================
assign da_unsigned = o_output + 16'sd32768;
assign da_round = {1'b0, da_unsigned} + 17'sd128;

always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        da_data <= 8'd0;
    end else begin
        da_data <= (da_round[16]) ? 8'd255 : da_round[15:8];
    end
end

assign da_clk = clk_100M;
assign da_data_out = da_data;

endmodule