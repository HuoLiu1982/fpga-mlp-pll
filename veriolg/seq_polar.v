////////////////////////////////////////////////////////////////////////////////
//
// Filename: 	seqpolar.v
//
// Project:	A collection of phase locked loop (PLL) related projects
//
// Purpose:	This is a rectangular to polar conversion routine based upon an
//		internal CORDIC implementation.  Basically, the input is
//	provided in i_xval and i_yval.  The internal CORDIC rotator will rotate
//	(i_xval, i_yval) until i_yval is approximately zero.  The resulting
//	xvalue and phase will be placed into o_xval and o_phase respectively.
//
//	This particular version of the polar to rectangular CORDIC converter
//	converter processes a somple one at a time.  It is completely
//	sequential, not parallel at all.
//
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
module	seqpolar (
		// {{{
		input	wire				i_clk, i_reset, i_stb,
		input	wire	signed	[15:0]	i_xval, i_yval,
		output	wire				o_busy,
		output	reg				o_done,
		output	reg	signed	[15:0]	o_mag,
		output	reg		[31:0]	o_phase
		// }}}
	);

	// Parameters
	// {{{
	parameter	IW = 16;	// The number of bits in our inputs
	parameter	OW = 16;	// The number of output bits to produce
	parameter	WW = 24;	// Our working bit-width
	parameter	PW = 32;	// Bits in our phase variables
	// }}}

	// First step: expand our input to our working width.
	// {{{
	// This is going to involve extending our input by one
	// (or more) bits in addition to adding any xtra bits on
	// bits on the right.  The one bit extra on the left is to
	// allow for any accumulation due to the cordic gain
	// within the algorithm.
	
	wire	signed [23:0]	e_xval, e_yval;
	
	assign	e_xval = { {2{i_xval[15]}}, i_xval, 6'b000000 };
	assign	e_yval = { {2{i_yval[15]}}, i_yval, 6'b000000 };
	// }}}

	// Declare variables for all of the separate stages
	// {{{
	reg	signed	[23:0]	xv, yv, prex, prey;
	reg		[31:0]	ph, preph;

	reg		idle, pre_valid;
	reg	[4:0]	state;

	wire		last_state;
	// }}}

	// First stage, map to within +/- 45 degrees
	// {{{
	always @(posedge i_clk) begin
		case({i_xval[15], i_yval[15]})
		2'b01: begin // Rotate by -315 degrees
			prex <=  e_xval - e_yval;
			prey <=  e_xval + e_yval;
			preph <= 32'he0000000;
			end
		2'b10: begin // Rotate by -135 degrees
			prex <= -e_xval + e_yval;
			prey <= -e_xval - e_yval;
			preph <= 32'h60000000;
			end
		2'b11: begin // Rotate by -225 degrees
			prex <= -e_xval - e_yval;
			prey <=  e_xval - e_yval;
			preph <= 32'ha0000000;
			end
		// 2'b00:
		default: begin // Rotate by -45 degrees
			prex <=  e_xval + e_yval;
			prey <= -e_xval + e_yval;
			preph <= 32'h20000000;
			end
		endcase
	end
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

	assign	last_state = (state >= 30);

	// idle
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			idle <= 1'b1;
		end else if (i_stb) begin
			idle <= 1'b0;
		end else if (last_state) begin
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

	// state
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			state <= 5'd0;
		end else if (idle) begin
			state <= 5'd0;
		end else if (last_state) begin
			state <= 5'd0;
		end else begin
			state <= state + 5'd1;
		end
	end
	// }}}

	// cangle -- table lookup using case statement
	// {{{
	always @(*) begin
		case (state)
			5'd0:  cangle = cordic_angle[0];
			5'd1:  cangle = cordic_angle[1];
			5'd2:  cangle = cordic_angle[2];
			5'd3:  cangle = cordic_angle[3];
			5'd4:  cangle = cordic_angle[4];
			5'd5:  cangle = cordic_angle[5];
			5'd6:  cangle = cordic_angle[6];
			5'd7:  cangle = cordic_angle[7];
			5'd8:  cangle = cordic_angle[8];
			5'd9:  cangle = cordic_angle[9];
			5'd10: cangle = cordic_angle[10];
			5'd11: cangle = cordic_angle[11];
			5'd12: cangle = cordic_angle[12];
			5'd13: cangle = cordic_angle[13];
			5'd14: cangle = cordic_angle[14];
			5'd15: cangle = cordic_angle[15];
			5'd16: cangle = cordic_angle[16];
			5'd17: cangle = cordic_angle[17];
			5'd18: cangle = cordic_angle[18];
			5'd19: cangle = cordic_angle[19];
			5'd20: cangle = cordic_angle[20];
			5'd21: cangle = cordic_angle[21];
			5'd22: cangle = cordic_angle[22];
			5'd23: cangle = cordic_angle[23];
			5'd24: cangle = cordic_angle[24];
			5'd25: cangle = cordic_angle[25];
			5'd26: cangle = cordic_angle[26];
			5'd27: cangle = cordic_angle[27];
			5'd28: cangle = cordic_angle[28];
			5'd29: cangle = cordic_angle[29];
			5'd30: cangle = cordic_angle[30];
			5'd31: cangle = cordic_angle[31];
			default: cangle = 32'h00000000;
		endcase
	end
	// }}}

	// Actual CORDIC rotation
	// {{{
	// Here's where we are going to put the actual CORDIC
	// rectangular to polar loop.  Everything up to this
	// point has simply been necessary preliminaries.
	always @(posedge i_clk)
	if (pre_valid)
	begin
		// {{{
		xv <= prex;
		yv <= prey;
		ph <= preph;
		// }}}
	end else if (yv[(WW-1)]) // Below the axis
	begin
		// {{{
		// If the vector is below the x-axis, rotate by
		// the CORDIC angle in a positive direction.
		xv <= xv - (yv>>>state);
		yv <= yv + (xv>>>state);
		ph <= ph - cangle;
		// }}}
	end else begin
		// {{{
		// On the other hand, if the vector is above the
		// x-axis, then rotate in the other direction
		xv <= xv + (yv>>>state);
		yv <= yv - (xv>>>state);
		ph <= ph + cangle;
		// }}}
	end
	// }}}

	// o_done
	// {{{
	always @(posedge i_clk) begin
		if (i_reset) begin
			o_done <= 1'b0;
		end else begin
			o_done <= (last_state);
		end
	end
	// }}}

	// }}}
	// Round our magnitude towards even
	// {{{
	wire	[(WW-1):0]	final_mag;

	assign	final_mag = xv + $signed({{(OW){1'b0}},
				xv[(WW-OW)],
				{(WW-OW-1){!xv[WW-OW]}} });

	// Output assignments: o_mag, o_phase
	// {{{
	always @(posedge i_clk) begin
		if (last_state) begin
			o_mag   <= final_mag[23:8];
			o_phase <= ph;
		end
	end
	// }}}

	assign	o_busy = !idle;
	// Make Verilator happy with pre_.val
	// {{{
	// verilator lint_off UNUSED
	wire	 unused_val;
	assign	unused_val = &{ 1'b0,  final_mag[WW-1],
			final_mag[(WW-OW-1):0] };
	// verilator lint_on UNUSED
	// }}}
endmodule