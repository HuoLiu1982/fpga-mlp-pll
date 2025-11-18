module sample #(
    parameter NUM = 16,            // 数据位宽
    parameter SAMPLE_POINTS = 64,   // 输出点数
    parameter MAX_SAMPLES = 1048576,   // 最大采样点数
    parameter ZERO_THRESHOLD = 7680   // 零点阈值 30LSB * 
)(
    input wire clk_100M,          // 100MHz主时钟
    input wire rst_n,               // 复位信号
    input wire enable,            // 使能信号
    input wire [NUM-1:0] adc_dai_A, // ADC是无符号数据,zero是2^(NUM-2)
    output reg done,              // 完成标志
    output reg [NUM-1:0] signal_tem, // 输出信号数据
    output reg [NUM/2 - 1 :0] signal_out//NUM/2
);

    // 内部寄存器
    reg [19:0] sample_counter = 0;
    reg [6:0] point_index = 0;
    reg [NUM-1:0] data_buffer [0:SAMPLE_POINTS-1];
    reg [NUM-1:0] prev_sample = 0;
    reg [19:0] prev_cnt = 0;
    reg [19:0] zero_array [0:6];
    reg [4:0] zero_cnt;
    reg [18:0] cycle_length = 0;
    reg [20:0] temp_interval;
    reg is_sample;
    
    // IIR滤波相关寄存器
    reg signed [NUM-1:0] signed_adc;
    reg signed [NUM-1:0] iir1_out, iir2_out;
    reg [NUM-1:0] filtered_adc;
    
    // Offset estimation registers
    reg signed [NUM-1:0] iir_max = -16'sd32768;
    reg signed [NUM-1:0] iir_min =  16'sd32767;
    reg signed [NUM-1:0] offset_est = 0;
    reg offset_ready = 0;
    // 状态机
    localparam IDLE = 3'b000;
    localparam OFFSET_CALC = 3'b001;  // 新增：偏移估计阶段
    localparam ZERO = 3'b010;
    localparam SAMPLE = 3'b011;
    localparam OUTPUT = 3'b100; // 注意：扩展为3位状态

    reg [2:0] state = IDLE;


    // ==================== 改进的过零点检测逻辑 ====================
    // 转换为有符号值进行比较
    wire signed [NUM-1:0] signed_prev = prev_sample - 16'd32768;
    wire signed [NUM-1:0] signed_current = filtered_adc - 16'd32768;
    
    // 信号状态检测
    wire is_above_threshold = (signed_prev > ZERO_THRESHOLD);
    wire is_below_threshold = (signed_prev < -ZERO_THRESHOLD);
    // 改进的过零点检测：只有从明确的正区域穿越到明确的负区域（或反之）才认为是真过零点
    reg was_above = 0;  // 之前是否在正区域
    reg was_below = 0;  // 之前是否在负区域
    
    // 真过零点条件：
    // 1. 之前明确在正区域，现在明确在负区域
    // 2. 之前明确在负区域，现在明确在正区域
    wire true_zero_cross = 
        (was_above && (signed_current < -ZERO_THRESHOLD)) ||  // 正->负穿越
        (was_below && (signed_current > ZERO_THRESHOLD));     // 负->正穿越
    always @(posedge clk_100M or negedge rst_n) begin
        if (!rst_n || true_zero_cross) begin
            was_above <= 0;
            was_below <= 0;
        end else if (enable) begin
            // 更新区域状态
            if (is_above_threshold) begin
                was_above <= 1;
                was_below <= 0;
            end else if (is_below_threshold) begin
                was_above <= 0;
                was_below <= 1;
            end
            // 如果在零点附近，保持之前的状态
        end
    end
    
    
    integer i;
    
    // ==================== IIR滤波处理 ====================
// 推荐写法：用乘法代替移位，意图更清晰，综合结果相同
parameter IW = 16;
parameter SCALE_FACTOR = 256;  // 2^8
parameter LARGE_NUM = 8;
reg signed [IW-1:0] ad_bias;
reg signed [IW+LARGE_NUM-1:0] amp;

    always @(posedge clk_100M) begin
        ad_bias <= adc_dai_A - 16'd32768;
        amp <= ad_bias * SCALE_FACTOR;  // 综合为算术左移，符号自动保留
        // 第三级：饱和处理
        if (amp > 23'sd32767) begin
            signed_adc <= 16'sd32767;
        end else if (amp < -23'sd32768) begin
            signed_adc <= -16'sd32768;
        end else begin
            signed_adc <= amp;
        end
    end


    
    // 两级IIR低通滤波
    always @(posedge clk_100M or negedge rst_n) begin
        if (!rst_n) begin
            iir1_out <= 0;
            iir2_out <= 0;
        end else if (enable) begin
            // 第一级：系数0.25
            iir1_out <= iir1_out - (iir1_out >>> 2) + (signed_adc >>> 2);
            // 第二级：系数0.125
            iir2_out <= iir2_out - (iir2_out >>> 3) + (iir1_out >>> 3);
        end
    end
    
    // ==================== 8点滑动平均滤波器，实际未生效 ====================
    // 滑动平均滤波器相关寄存器
    reg [NUM-1:0] moving_avg_buffer [0:7];
    reg [2:0] avg_index = 0;
    reg [NUM+2:0] moving_sum;
    always @(posedge clk_100M or negedge rst_n) begin
        if (!rst_n) begin
            avg_index <= 0;
            moving_sum <= 0;
            for (i = 0; i < 8; i = i + 1) begin
                moving_avg_buffer[i] <= 16'd32768;
            end
            filtered_adc <= 0;
        end else if (enable) begin
            moving_avg_buffer[avg_index] <= iir2_out + 16'd32768;
            
            if (avg_index == 0) begin
                moving_sum <= moving_sum - moving_avg_buffer[7] + moving_avg_buffer[0];
            end else begin
                moving_sum <= moving_sum - moving_avg_buffer[avg_index-1] + moving_avg_buffer[avg_index];
            end            
            avg_index <= (avg_index == 3'd7) ? 3'd0 : avg_index + 3'd1;
            //filtered_adc <= $signed(signed_adc) + 16'sd32768;
            // 应用偏移校正
            if (offset_ready) begin
                filtered_adc <= $signed(iir2_out - offset_est) + 16'sd32768;
            end else begin
                filtered_adc <= $signed(iir2_out) + 16'sd32768;
            end
            //moving_sum >>> 3;

        end
    end
/*
    // ==================== 8点滑动平均滤波器，实际生效 ====================
    // 滑动平均滤波器相关寄存器
    reg signed [NUM-1:0] moving_avg_buffer [0:7];
    reg [2:0] avg_index = 0;
    reg signed [NUM+3:0] moving_sum;
always @(posedge clk_100M or negedge rst_n) begin
    if (!rst_n) begin
        avg_index <= 0;
        moving_sum <= 0;
        for (i = 0; i < 8; i = i + 1)
            moving_avg_buffer[i] <= 0;
        filtered_adc <= 0;
    end
    else if (enable) begin
        // 1. 减掉最老样本
        moving_sum <= moving_sum - moving_avg_buffer[avg_index];

        // 2. 写入新样本
        moving_avg_buffer[avg_index] <= iir2_out;

        // 3. 加上新样本
        moving_sum <= moving_sum + iir2_out;

        // 4. 指针循环
        avg_index <= (avg_index == 3'd7) ? 3'd0 : avg_index + 1'd1;
        
        
        // 5. 输出 8 点平均（带偏移校正）
        if (offset_ready)
            filtered_adc <= $signed((moving_sum - offset_est) >>> 3) 
                            + 16'd32768;
        else
            filtered_adc <= iir2_out + 16'sd32768;
    end
end
*/ 
    // ==================== 主状态机 ====================
    always @(posedge clk_100M or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            sample_counter <= 0;
            point_index <= 0;
            prev_sample <= 0;
            zero_cnt <= 0;
            prev_cnt <= 0;
            cycle_length <= 0;
            temp_interval <= 0;
            is_sample <= 0;
            iir_max <= -16'sd32768;
            iir_min <=  16'sd32767;
            offset_ready <= 0;
        end else begin
            prev_sample <= filtered_adc;
            
            case(state)
                IDLE: begin
                    done <= 0;
                    sample_counter <= 0;
                    point_index <= 0;
                    prev_cnt <= 0;
                    zero_cnt <= 0;
                    cycle_length <= 0;
                    temp_interval <= 0;
                    is_sample <= 0;
                    iir_max <= -16'sd32768;
                    iir_min <=  16'sd32767;
                    offset_ready <= 0;
                    if (enable) begin
                        state <= OFFSET_CALC;
                    end
                end
                
                OFFSET_CALC: begin
                    if (enable) begin
                        if (iir2_out > iir_max) iir_max <= iir2_out;
                        if (iir2_out < iir_min) iir_min <= iir2_out;
                        sample_counter <= sample_counter + 1;
                        
                        if (sample_counter == 20'd131072) begin//注意修改为131072
                            offset_est <= (iir_max + iir_min) >>> 1;
                            offset_ready <= 1;
                            sample_counter <= 0;
                            state <= ZERO;
                        end
                    end else begin
                        state <= IDLE;
                    end
                end
                
                ZERO: begin
                    if (sample_counter >= MAX_SAMPLES) begin
                        state <= IDLE;
                    end
                    if (enable) begin
                        if (true_zero_cross) begin
                            prev_cnt <= sample_counter;
                            sample_counter <= 1;
                        end else begin
                            sample_counter <= sample_counter + 1;
                        end
                        
                        if(sample_counter == 1) begin
                            zero_array[zero_cnt] <= prev_cnt;
                            zero_cnt <= zero_cnt + 1;
                        end
                        
                        if(zero_cnt == 5'd7) begin
                            cycle_length <= (zero_array[3] + zero_array[4] + 
                                           zero_array[5] + zero_array[6]) >> 1;
                            is_sample <= 1;
                        end
                        
                        if(is_sample) begin
                            temp_interval <= (cycle_length >> 6) + 18'd1;//尝试只加1
                            state <= SAMPLE;
                            sample_counter <= 0;
                        end					
                    end else begin
                        state <= IDLE;
                    end
                end
                
                SAMPLE: begin					
                    if ((sample_counter % temp_interval) == 0 
                        && point_index < SAMPLE_POINTS) begin
                        data_buffer[point_index] <= filtered_adc;
                        point_index <= point_index + 1;
                    end
                    
                    if (point_index >= SAMPLE_POINTS) begin
                        state <= OUTPUT;
                        point_index <= 0;
                    end
                    sample_counter <= sample_counter + 1;
                end
                
                OUTPUT: begin
                    if (point_index < SAMPLE_POINTS) begin
                        done <= 1;
                        signal_tem <= data_buffer[point_index];//去除+128
                        signal_out <= signal_tem[NUM-1 : NUM/2];
                        point_index <= point_index + 1;
                    end else begin
                        done <= 0;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule