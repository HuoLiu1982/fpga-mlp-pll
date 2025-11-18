////////////////////////////////////////////////////////////////////////////////
//
// Filename: 	dblcordicpll.v
//
// Project:	A collection of phase locked loop (PLL) related projects
//
// Purpose:	A high precision PLL based upon using two internal CORDICs:
//		One for multiplying the incoming signal by sine/cosine, followed
//	by a filter and then the second CORDIC for measuring the resulting
//	phase.
//
//	The delay of this digital phase lock loop (DPLL) is derived from the
//	delays of the two cordics.  The sample rate *must* remain low enough
//	that the two CORDICs have time to calculate their answer before the
//	next sample comes in.
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

`default_nettype	none

module	dblcordicpll (
		// {{{
		input	wire				i_clk,
		input	wire				i_reset,  //   ? ¦Ë ? 
		// Frequency control
		input	wire				i_ld,
		input	wire		[31:0]	i_step,
		// Incoming signal
		input	wire				i_ce,
		input	wire	signed	[15:0]	i_input,
		// Tracking control
		input	wire	[4:0]			i_lgcoeff,
		// Outgoiing error
		output	wire	[15:0]			o_err,
		output	wire	[31:0]			o_phase
		// }}}
	);

	// Parameters
	// {{{
	parameter	IW = 16; // input width
	parameter	PW = 32; // phase width in bits
	parameter	OW = 16; // output width
	parameter	OPT_TRACK_FREQUENCY = 1'b1;
	parameter	OPT_FILTER = 1'b1;
	// }}}

	// Internal register/wire declarations
	// {{{
	reg	[31:0]		r_phase, r_step;

	wire			pm_busy, pm_done;
	wire	signed	[15:0]	pm_sin, pm_cos;
	//
	wire			fil_ce;
	reg	signed	[15:0]	fil_sin, fil_cos;
	//
	wire			pd_busy, pd_done;
	wire	[15:0]		pd_mag;
	wire	signed [31:0]	pd_phase;
	reg	[4:0]		log_gamma;
	reg	signed [31:0]	phase_correction;

	// ?   ?  ¦Ë ? 
	wire s_reset = i_reset;

	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Multiply
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
	seqcordic phasemultiply (
		.i_clk(i_clk),
		.i_reset(s_reset),
		.i_stb(i_ce),
		.i_xval(16'd0),
		.i_yval(i_input),
		.i_phase(~r_phase),  // ? ?    ?    ~      
		.o_busy(pm_busy),
		.o_done(pm_done),
		.o_xval(pm_cos),
		.o_yval(pm_sin)
	);
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Filter
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
	generate if (OPT_FILTER && OPT_TRACK_FREQUENCY)
	begin : TRACKING_SIN_COS
		// {{{
		reg		[2:0]		shift_ce;
		reg	signed	[15:0]	diff_sin, diff_cos,
						tripl_sin, tripl_cos;
		reg	[5:0]			log_alpha;

		always @(*) begin
			log_alpha = i_lgcoeff - 1;
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				shift_ce <= 3'b000;
			end else begin
				shift_ce <= { shift_ce[1:0], pm_done };
			end
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				diff_sin <= 16'd0;
				diff_cos <= 16'd0;
			end else if (pm_done) begin
				diff_sin <= pm_sin - fil_sin;
				diff_cos <= pm_cos - fil_cos;
			end
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				tripl_sin <= 16'd0;
				tripl_cos <= 16'd0;
			end else if (shift_ce[0]) begin
				tripl_sin <= diff_sin + (diff_sin >>> 1);
				tripl_cos <= diff_cos + (diff_cos >>> 1);
			end
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				fil_sin <= 16'd0;
				fil_cos <= 16'd0;
			end else if (shift_ce[1]) begin
				fil_sin <= fil_sin + (tripl_sin >>> log_alpha);
				fil_cos <= fil_cos + (tripl_cos >>> log_alpha);
			end
		end

		assign	fil_ce = shift_ce[2];
		// }}}
	end else if (OPT_FILTER)
	begin : FILTER_SIN_COS
		// {{{
		reg		[1:0]		shift_ce;
		reg	signed	[15:0]	diff_sin, diff_cos;
		reg	[5:0]			log_alpha;

		always @(*) begin
			log_alpha = i_lgcoeff - 2;
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				shift_ce <= 2'b00;
			end else begin
				shift_ce <= { shift_ce[0], pm_done };
			end
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				diff_sin <= 16'd0;
				diff_cos <= 16'd0;
			end else if (pm_done) begin
				diff_sin <= pm_sin - fil_sin;
				diff_cos <= pm_cos - fil_cos;
			end
		end

		// ?    ?       ? 
		wire signed [16:0] diff_sin_rnd = diff_sin + (1 << (log_alpha - 1));
		wire signed [16:0] diff_cos_rnd = diff_cos + (1 << (log_alpha - 1));

		always @(posedge i_clk) begin
			if (s_reset) begin
				fil_sin <= 16'd0;
				fil_cos <= 16'd0;
			end else if (shift_ce[0]) begin
				fil_sin <= fil_sin + (diff_sin_rnd >>> log_alpha);
				fil_cos <= fil_cos + (diff_cos_rnd >>> log_alpha);
			end
		end

		assign	fil_ce = shift_ce[1];
		// }}}
	end else begin : OTHER
		// {{{
		assign	fil_ce  = pm_done;
		
		always @(posedge i_clk) begin
			if (s_reset) begin
				fil_sin <= 16'd0;
				fil_cos <= 16'd0;
			end else begin
				fil_sin <= pm_sin;
				fil_cos <= pm_cos;
			end
		end
		// }}}
	end endgenerate
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Atan2 phase detect
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	seqpolar phasedet (
		.i_clk(i_clk),
		.i_reset(s_reset),
		.i_stb(fil_ce),
		.i_xval(fil_cos),
		.i_yval(fil_sin),
		.o_busy(pd_busy),
		.o_done(pd_done),
		.o_mag(pd_mag),
		.o_phase(pd_phase)
	);

	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Tracking loops
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
	always @(*) begin
		log_gamma = i_lgcoeff;
	end

	// Track frequency
	generate if (OPT_TRACK_FREQUENCY)
	begin : FREQ_CORRECTION
		// {{{
		reg	[5:0]			log_beta;
		reg	signed [31:0]		freq_correction;

		always @(*) begin
			// beta = gamma ^2 / 4
			log_beta = { i_lgcoeff, 1'b0 } + 2;
		end

		always @(*) begin
			freq_correction = pd_phase >>> log_beta;
		end

		always @(posedge i_clk) begin
			if (s_reset) begin
				r_step <= 32'd0;
			end else if (i_ld) begin
				r_step <= i_step;
			end else if (pd_done) begin
				r_step <= r_step - freq_correction;
			end
		end
		// }}}
	end else begin : PHASE_CORRECTION_ONLY
		// {{{
		always @(posedge i_clk) begin
			if (s_reset) begin
				r_step <= 32'd0;
			end else if (i_ld) begin
				r_step <= i_step;
			end
		end
		// }}}
	end endgenerate

	assign	o_err = pd_phase[31:16];

	// Track phase
	// {{{
	always @(*) begin
		phase_correction = pd_phase >>> log_gamma;
	end

	always @(posedge i_clk) begin
		if (s_reset) begin
			r_phase <= 32'd0;
		end else if (pd_done) begin
			r_phase <= r_phase + r_step - phase_correction;
		end
	end
	// }}}
	// }}}

	// Make Verilator happy
	// {{{
	// verilator lint_off UNUSED
	wire	unused;
	assign	unused = &{ 1'b0, pm_busy, pd_busy, pd_mag };
	// verilator lint_on  UNUSED
	// }}}
	
	assign o_phase = r_phase;
endmodule