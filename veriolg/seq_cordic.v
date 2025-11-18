////////////////////////////////////////////////////////////////////////////////
//
// Filename: 	seqcordic.v
//
// Project:	A collection of phase locked loop (PLL) related projects
//
// Purpose:	This file executes a vector rotation on the values
//		(i_xval, i_yval).  This vector is rotated left by
//	i_phase.  i_phase is given by the angle, in radians, multiplied by
//	2^32/(2pi).  In that fashion, a two pi value is zero just as a zero
//	angle is zero.
//
//	This particular version of the CORDIC processes one value at a
//	time in a sequential, vs pipelined or parallel, fashion.
//
// Creator:	Dan Gisselquist, Ph.D.
//		Gisselquist Technology, LLC
//
////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2020-2024, Gisselquist Technology, LLC
//
// This program is free software (firmware): you can redistribute it and/or
// modify it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or (at
// your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTIBILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/> for a copy.
//
// License:	GPL, v3, as defined and found on www.gnu.org,
//		http://www.gnu.org/licenses/gpl.html
//
////////////////////////////////////////////////////////////////////////////////

module	seqcordic (
		// {{{
		input	wire				i_clk, i_reset, i_stb,
		input	wire	signed	[15:0]	i_xval, i_yval,
		input	wire		[31:0]	i_phase,
		output	wire				o_busy,
		output	reg				o_done,
		output	reg	signed	[15:0]	o_xval, o_yval
		// }}}
	);

	// Parameters fixed by the core generator
	// {{{
	// These parameters are fixed by the core generator. They
	// have been used in the definitions of internal constants,
	// so they can't really be changed here.
	parameter	IW = 16;	// The number of bits in our inputs
	parameter	OW = 16;	// The number of output bits to produce
	parameter	WW = 19;	// Our working bit-width
	parameter	PW = 32;	// Bits in our phase variables
	parameter	itercnt = 19;//迭代次数19-1
	// }}}

	// First step: expand our input to our working width.
	// {{{
	// This is going to involve extending our input by one
	// (or more) bits in addition to adding any xtra bits on
	// bits on the right. The one bit extra on the left is to
	// allow for any accumulation due to the cordic gain
	// within the algorithm.
	
	wire	signed [18:0]	e_xval, e_yval;
	
	assign	e_xval = { {i_xval[15]}, i_xval, 2'b00 };
	assign	e_yval = { {i_yval[15]}, i_yval, 2'b00 };
	// }}}

	// Declare variables for all of the separate stages
	// {{{
	reg	signed	[18:0]	xv, prex, yv, prey;
	reg		[31:0]	ph, preph;
	reg				idle, pre_valid;
	reg		[4:0]		state;

	// }}}

	// Cordic angle table - defined as wires for Verilog-2001 compatibility
	// {{{
	wire [31:0] cordic_angle [0:31];
	
	assign cordic_angle[ 0] = 32'h12e4051d; //  26.565051 deg
	assign cordic_angle[ 1] = 32'h09fb385b; //  14.036243 deg
	assign cordic_angle[ 2] = 32'h051111d4; //   7.125016 deg
	assign cordic_angle[ 3] = 32'h028b0d43; //   3.576334 deg
	assign cordic_angle[ 4] = 32'h0145d7e1; //   1.789911 deg
	assign cordic_angle[ 5] = 32'h00a2f61e; //   0.895174 deg
	assign cordic_angle[ 6] = 32'h00517c55; //   0.447614 deg
	assign cordic_angle[ 7] = 32'h0028be53; //   0.223811 deg
	assign cordic_angle[ 8] = 32'h00145f2e; //   0.111906 deg
	assign cordic_angle[ 9] = 32'h000a2f98; //   0.055953 deg
	assign cordic_angle[10] = 32'h000517cc; //   0.027976 deg
	assign cordic_angle[11] = 32'h00028be6; //   0.013988 deg
	assign cordic_angle[12] = 32'h000145f3; //   0.006994 deg
	assign cordic_angle[13] = 32'h0000a2f9; //   0.003497 deg
	assign cordic_angle[14] = 32'h0000517c; //   0.001749 deg
	assign cordic_angle[15] = 32'h000028be; //   0.000874 deg
	assign cordic_angle[16] = 32'h0000145f; //   0.000437 deg
	assign cordic_angle[17] = 32'h00000a2f; //   0.000219 deg
	assign cordic_angle[18] = 32'h00000517; //   0.000109 deg
	assign cordic_angle[19] = 32'h0000028b; //   0.000055 deg
	assign cordic_angle[20] = 32'h00000145; //   0.000027 deg
	assign cordic_angle[21] = 32'h000000a2; //   0.000014 deg
	assign cordic_angle[22] = 32'h00000051; //   0.000007 deg
	assign cordic_angle[23] = 32'h00000028; //   0.000003 deg
	assign cordic_angle[24] = 32'h00000014; //   0.000002 deg
	assign cordic_angle[25] = 32'h0000000a; //   0.000001 deg
	assign cordic_angle[26] = 32'h00000005; //   0.000000 deg
	assign cordic_angle[27] = 32'h00000002; //   0.000000 deg
	assign cordic_angle[28] = 32'h00000001; //   0.000000 deg
	assign cordic_angle[29] = 32'h00000000; //   0.000000 deg
	assign cordic_angle[30] = 32'h00000000; //   0.000000 deg
	assign cordic_angle[31] = 32'h00000000; //   0.000000 deg
	
	reg [31:0] cangle;
	// }}}

	// First step, get rid of all but the last 45 degrees
	// {{{
	// The resulting phase needs to be between -45 and 45 degrees
	// but in units of normalized phase
	always @(posedge i_clk) begin
		case(i_phase[31:29])
		3'b000: begin	// 0 .. 45, No change
			prex  <=  e_xval;
			prey  <=  e_yval;
			preph <= i_phase;
			end
		3'b001: begin	// 45 .. 90
			prex  <= -e_yval;
			prey  <=  e_xval;
			preph <= i_phase - 32'h40000000;
			end
		3'b010: begin	// 90 .. 135
			prex  <= -e_yval;
			prey  <=  e_xval;
			preph <= i_phase - 32'h40000000;
			end
		3'b011: begin	// 135 .. 180
			prex  <= -e_xval;
			prey  <= -e_yval;
			preph <= i_phase - 32'h80000000;
			end
		3'b100: begin	// 180 .. 225
			prex  <= -e_xval;
			prey  <= -e_yval;
			preph <= i_phase - 32'h80000000;
			end
		3'b101: begin	// 225 .. 270
			prex  <=  e_yval;
			prey  <= -e_xval;
			preph <= i_phase - 32'hc0000000;
			end
		3'b110: begin	// 270 .. 315
			prex  <=  e_yval;
			prey  <= -e_xval;
			preph <= i_phase - 32'hc0000000;
			end
		3'b111: begin	// 315 .. 360, No change
			prex  <=  e_xval;
			prey  <=  e_yval;
			preph <= i_phase;
			end
		endcase
	end
	// }}}

	// idle
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			idle <= 1'b1;
		end else if (i_stb) begin
			idle <= 1'b0;
		end else if (state == itercnt - 1) begin
			idle <= 1'b1;
		end
	end
	// }}}

	// pre_valid
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			pre_valid <= 1'b0;
		end else begin
			pre_valid <= (i_stb) && (idle);
		end
	end
	// }}}

// 关键修复：cangle使用寄存器查找，否则会导致波形变形（原组合逻辑）
    always @(posedge i_clk) begin
        case (state)
            5'd0:  cangle <= cordic_angle[0];
            5'd1:  cangle <= cordic_angle[1];
            5'd2:  cangle <= cordic_angle[2];
            5'd3:  cangle <= cordic_angle[3];
            5'd4:  cangle <= cordic_angle[4];
            5'd5:  cangle <= cordic_angle[5];
            5'd6:  cangle <= cordic_angle[6];
            5'd7:  cangle <= cordic_angle[7];
            5'd8:  cangle <= cordic_angle[8];
            5'd9:  cangle <= cordic_angle[9];
            5'd10: cangle <= cordic_angle[10];
            5'd11: cangle <= cordic_angle[11];
            5'd12: cangle <= cordic_angle[12];
            5'd13: cangle <= cordic_angle[13];
            5'd14: cangle <= cordic_angle[14];
            5'd15: cangle <= cordic_angle[15];
            5'd16: cangle <= cordic_angle[16];
            5'd17: cangle <= cordic_angle[17];
            5'd18: cangle <= cordic_angle[18];
            default: cangle <= 32'h00000000;
        endcase
    end
    

	// state
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			state <= 5'd0;
		end else if (idle) begin
			state <= 5'd0;
		end else if (state == itercnt - 1) begin
			state <= 5'd0;
		end else begin
			state <= state + 5'd1;
		end
	end
	// }}}

	// CORDIC rotations
	// {{{
	// Here's where we are going to put the actual CORDIC
	// we've been studying and discussing.  Everything up to
	// this point has simply been necessary preliminaries.
	always @(posedge i_clk)
	if (pre_valid)
	begin
		// {{{
		xv <= prex;
		yv <= prey;
		ph <= preph;
		// }}}
	end else if (ph[PW-1])
	begin
		// {{{
		xv <= xv + (yv >>> state);
		yv <= yv - (xv >>> state);
		ph <= ph + (cangle);
		// }}}
	end else begin
		// {{{
		xv <= xv - (yv >>> state);
		yv <= yv + (xv >>> state);
		ph <= ph - (cangle);
		// }}}
	end
	// }}}

	// Round our result towards even
	// {{{
	wire	[(WW-1):0]	final_xv, final_yv;

	assign	final_xv = xv + $signed({{(OW){1'b0}},
				xv[(WW-OW)],
				{(WW-OW-1){!xv[WW-OW]}} });
	assign	final_yv = yv + $signed({{(OW){1'b0}},
				yv[(WW-OW)],
				{(WW-OW-1){!yv[WW-OW]}} });
	// }}}

	// o_done
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			o_done <= 1'b0;
		end else begin
			o_done <= (state >= itercnt - 1);
		end
	end
	// }}}

	// Output assignments: o_xval, o_yval
	// {{{
	always @(posedge i_clk) begin
		if (state >= itercnt - 1) begin
			o_xval <= final_xv[18:3];
			o_yval <= final_yv[18:3];
		end
	end
	// }}}

	assign	o_busy = !idle;

endmodule