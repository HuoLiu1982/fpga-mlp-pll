`timescale 1ns / 1ps

// MLP LUTs for FPGA Implementation
// Auto-generated from PyTorch model
// Q10.22 fixed-point format

module layer1_bias_lut(
    input [4:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        5'd0: data = 32'hffff6de8;
        5'd1: data = 32'hffff05d5;
        5'd2: data = 32'hfffe2149;
        5'd3: data = 32'hfffffd24;
        5'd4: data = 32'hffff4a2e;
        5'd5: data = 32'h000136cc;
        5'd6: data = 32'hfffef45a;
        5'd7: data = 32'h00024ca7;
        5'd8: data = 32'hfffe403f;
        5'd9: data = 32'h000cbbde;
        5'd10: data = 32'hfffe3e9b;
        5'd11: data = 32'hfffa6e7f;
        5'd12: data = 32'hfffce9be;
        5'd13: data = 32'h00018d1d;
        5'd14: data = 32'hfffefc5e;
        5'd15: data = 32'h000290cf;
        5'd16: data = 32'h0001a635;
        5'd17: data = 32'h0004bc99;
        5'd18: data = 32'hfffab09c;
        5'd19: data = 32'hfffaae16;
        5'd20: data = 32'h000eda46;
        5'd21: data = 32'h00007b1a;
        5'd22: data = 32'hfffb2962;
        5'd23: data = 32'hffff78fb;
        5'd24: data = 32'h000733ae;
        5'd25: data = 32'h0003db1e;
        5'd26: data = 32'h0002279c;
        5'd27: data = 32'hffef194a;
        5'd28: data = 32'h0006eb11;
        5'd29: data = 32'h0005a2e4;
        5'd30: data = 32'h000d43e1;
        5'd31: data = 32'hfffbd5d4;
        default: data = 32'h00000000;
    endcase
end

endmodule

module layer2_weight_lut(
    input [9:0] addr,  // 16*32=512个权重
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        10'd0: data = 32'hffda176d;
        10'd1: data = 32'h00041d77;
        10'd2: data = 32'h000fbfa0;
        10'd3: data = 32'h000cca51;
        10'd4: data = 32'h0015360c;
        10'd5: data = 32'hff992f3b;
        10'd6: data = 32'hffebcfbf;
        10'd7: data = 32'h00189da0;
        10'd8: data = 32'h0008f6ca;
        10'd9: data = 32'hffeda706;
        10'd10: data = 32'h0012f0a0;
        10'd11: data = 32'hffbae180;
        10'd12: data = 32'h000a29a9;
        10'd13: data = 32'h00145570;
        10'd14: data = 32'h0026d876;
        10'd15: data = 32'h00092add;
        10'd16: data = 32'h0006239f;
        10'd17: data = 32'hffcc1f51;
        10'd18: data = 32'hffe33f5f;
        10'd19: data = 32'h002a8a80;
        10'd20: data = 32'h0004dcd3;
        10'd21: data = 32'h0018def3;
        10'd22: data = 32'h00150436;
        10'd23: data = 32'hfffec631;
        10'd24: data = 32'hffe4f465;
        10'd25: data = 32'hffe425d9;
        10'd26: data = 32'h00079cf4;
        10'd27: data = 32'h0001b52d;
        10'd28: data = 32'h000907c9;
        10'd29: data = 32'h00057fb4;
        10'd30: data = 32'hffe61c8f;
        10'd31: data = 32'h00067d36;
        10'd32: data = 32'hfffbd612;
        10'd33: data = 32'h000188ff;
        10'd34: data = 32'hffe3f125;
        10'd35: data = 32'hffd3a1c8;
        10'd36: data = 32'hffe2ea89;
        10'd37: data = 32'h000202ae;
        10'd38: data = 32'h0012020c;
        10'd39: data = 32'hffe2915f;
        10'd40: data = 32'hfffca6d8;
        10'd41: data = 32'h00178209;
        10'd42: data = 32'hffee5e79;
        10'd43: data = 32'hfffded96;
        10'd44: data = 32'hffdbb7fd;
        10'd45: data = 32'hffeaeb6b;
        10'd46: data = 32'hffec577e;
        10'd47: data = 32'hffe916ce;
        10'd48: data = 32'h0000dfea;
        10'd49: data = 32'hffd5fb24;
        10'd50: data = 32'h001373b6;
        10'd51: data = 32'h003854d4;
        10'd52: data = 32'hfffefbaf;
        10'd53: data = 32'hffe7f01c;
        10'd54: data = 32'h0046f142;
        10'd55: data = 32'h00085236;
        10'd56: data = 32'h0018392c;
        10'd57: data = 32'h0010de38;
        10'd58: data = 32'hffef6a43;
        10'd59: data = 32'hfffe3b0d;
        10'd60: data = 32'hfffdf5f1;
        10'd61: data = 32'h000008cd;
        10'd62: data = 32'h0011a830;
        10'd63: data = 32'hfffe7bd4;
        10'd64: data = 32'hffe6cfed;
        10'd65: data = 32'hfffd6fb8;
        10'd66: data = 32'h001b5db9;
        10'd67: data = 32'h00049f18;
        10'd68: data = 32'h001f187b;
        10'd69: data = 32'hffff503a;
        10'd70: data = 32'h0013db34;
        10'd71: data = 32'h00145a02;
        10'd72: data = 32'hffd1bba6;
        10'd73: data = 32'h001ec463;
        10'd74: data = 32'h000f09d9;
        10'd75: data = 32'h001571f0;
        10'd76: data = 32'hfff8e605;
        10'd77: data = 32'h0008f091;
        10'd78: data = 32'hfff37670;
        10'd79: data = 32'h000735a5;
        10'd80: data = 32'hffc57c27;
        10'd81: data = 32'h0016d509;
        10'd82: data = 32'h000aa03c;
        10'd83: data = 32'hfff7fb86;
        10'd84: data = 32'hffc8fe79;
        10'd85: data = 32'h001b6b37;
        10'd86: data = 32'hffff1c0a;
        10'd87: data = 32'h0006a107;
        10'd88: data = 32'hfffa9484;
        10'd89: data = 32'h000377ba;
        10'd90: data = 32'hfff3b5d5;
        10'd91: data = 32'hffe74241;
        10'd92: data = 32'hffc4973b;
        10'd93: data = 32'hffc7190f;
        10'd94: data = 32'hfff17e26;
        10'd95: data = 32'hffd01bf2;
        10'd96: data = 32'hffc4017f;
        10'd97: data = 32'hffb15caf;
        10'd98: data = 32'h0019d4bd;
        10'd99: data = 32'h000f1037;
        10'd100: data = 32'h000fde3f;
        10'd101: data = 32'hfffa7677;
        10'd102: data = 32'hfff3401d;
        10'd103: data = 32'h001b884b;
        10'd104: data = 32'hfffd5d99;
        10'd105: data = 32'hfff15f76;
        10'd106: data = 32'h00106c2e;
        10'd107: data = 32'hfffdf5d5;
        10'd108: data = 32'hfff17641;
        10'd109: data = 32'h000cedfe;
        10'd110: data = 32'h00239b1a;
        10'd111: data = 32'h001f0420;
        10'd112: data = 32'hffff8ae1;
        10'd113: data = 32'hffc2b184;
        10'd114: data = 32'hfff018b7;
        10'd115: data = 32'hfffca571;
        10'd116: data = 32'hfffe81c6;
        10'd117: data = 32'h002387a5;
        10'd118: data = 32'h0000a34a;
        10'd119: data = 32'h0019b223;
        10'd120: data = 32'hfff2b7d1;
        10'd121: data = 32'hfff3cf6e;
        10'd122: data = 32'h001a7e6a;
        10'd123: data = 32'hfffe5ad9;
        10'd124: data = 32'hffff418e;
        10'd125: data = 32'h00002a64;
        10'd126: data = 32'hfff06495;
        10'd127: data = 32'h0003ed6e;
        10'd128: data = 32'hfffd94d9;
        10'd129: data = 32'hfffd5965;
        10'd130: data = 32'hfff75487;
        10'd131: data = 32'hfff85196;
        10'd132: data = 32'hfff51e77;
        10'd133: data = 32'hfffd7e12;
        10'd134: data = 32'hffffb632;
        10'd135: data = 32'hfff7784e;
        10'd136: data = 32'h004356e6;
        10'd137: data = 32'hffffa420;
        10'd138: data = 32'hfffbaf6d;
        10'd139: data = 32'h00039a5e;
        10'd140: data = 32'h00403907;
        10'd141: data = 32'hffff2cfb;
        10'd142: data = 32'hfffe66b9;
        10'd143: data = 32'hfff8d068;
        10'd144: data = 32'h0046569e;
        10'd145: data = 32'hfff97bfc;
        10'd146: data = 32'hfff7fd7d;
        10'd147: data = 32'hffffa7b3;
        10'd148: data = 32'hffec2abc;
        10'd149: data = 32'hfff8d16a;
        10'd150: data = 32'hfffc6e09;
        10'd151: data = 32'hfffdea52;
        10'd152: data = 32'hffff0e7e;
        10'd153: data = 32'hfffafad5;
        10'd154: data = 32'hfffe6e10;
        10'd155: data = 32'hffe4a253;
        10'd156: data = 32'h005665c0;
        10'd157: data = 32'h003c190f;
        10'd158: data = 32'hffffb658;
        10'd159: data = 32'h000f7be0;
        10'd160: data = 32'h000aeabb;
        10'd161: data = 32'h0001e8e1;
        10'd162: data = 32'hfffe20d3;
        10'd163: data = 32'h001e728b;
        10'd164: data = 32'h00421b45;
        10'd165: data = 32'hfffc123d;
        10'd166: data = 32'hffcb55d6;
        10'd167: data = 32'h00186c2b;
        10'd168: data = 32'hffe50519;
        10'd169: data = 32'hffd1fde2;
        10'd170: data = 32'h0016581a;
        10'd171: data = 32'hfffec0f8;
        10'd172: data = 32'hfff2d73d;
        10'd173: data = 32'hfffafd50;
        10'd174: data = 32'h0007cc8c;
        10'd175: data = 32'h00229a71;
        10'd176: data = 32'hffe250c5;
        10'd177: data = 32'h00039cb7;
        10'd178: data = 32'hffbef344;
        10'd179: data = 32'h0008795e;
        10'd180: data = 32'hffe20ec4;
        10'd181: data = 32'h000c31db;
        10'd182: data = 32'h0010328d;
        10'd183: data = 32'h0007b802;
        10'd184: data = 32'hffd5675f;
        10'd185: data = 32'hffcb8707;
        10'd186: data = 32'h001113d0;
        10'd187: data = 32'hffecced7;
        10'd188: data = 32'hffd1ced6;
        10'd189: data = 32'hffe0422d;
        10'd190: data = 32'hffd1348a;
        10'd191: data = 32'hfff25126;
        10'd192: data = 32'h00082bac;
        10'd193: data = 32'hfff0ad92;
        10'd194: data = 32'h0005fd1e;
        10'd195: data = 32'hfffa5b76;
        10'd196: data = 32'hfff5327c;
        10'd197: data = 32'h00306b5a;
        10'd198: data = 32'hffeb0ff7;
        10'd199: data = 32'hffc90c57;
        10'd200: data = 32'hfff78791;
        10'd201: data = 32'hfff22ca6;
        10'd202: data = 32'hfff3a979;
        10'd203: data = 32'hfff7602c;
        10'd204: data = 32'hfff991cc;
        10'd205: data = 32'hffc7b164;
        10'd206: data = 32'hffbe13a8;
        10'd207: data = 32'hfff5b6e3;
        10'd208: data = 32'hfff7836c;
        10'd209: data = 32'h0007565b;
        10'd210: data = 32'hffec1fec;
        10'd211: data = 32'h00061bfe;
        10'd212: data = 32'hfff99ef7;
        10'd213: data = 32'hffbaf4dc;
        10'd214: data = 32'hfffd208a;
        10'd215: data = 32'hffc5d7ef;
        10'd216: data = 32'hffec468d;
        10'd217: data = 32'hffea236e;
        10'd218: data = 32'hffb11f72;
        10'd219: data = 32'hfffa50f6;
        10'd220: data = 32'hfff2bef9;
        10'd221: data = 32'hfff413b2;
        10'd222: data = 32'hffe90359;
        10'd223: data = 32'hfff7e10e;
        10'd224: data = 32'hfffc8c8e;
        10'd225: data = 32'hfffbfac6;
        10'd226: data = 32'hfff6cf37;
        10'd227: data = 32'hfffae790;
        10'd228: data = 32'hffeff84b;
        10'd229: data = 32'hfff8a3d5;
        10'd230: data = 32'hffdf4f49;
        10'd231: data = 32'hfff76296;
        10'd232: data = 32'h0035c56a;
        10'd233: data = 32'hffe0c0f4;
        10'd234: data = 32'hfffb6643;
        10'd235: data = 32'h001c5742;
        10'd236: data = 32'h00075c6a;
        10'd237: data = 32'hfffd6477;
        10'd238: data = 32'hfffcc5a7;
        10'd239: data = 32'hfff90a1d;
        10'd240: data = 32'h002d84d2;
        10'd241: data = 32'hfffa6940;
        10'd242: data = 32'hffdae916;
        10'd243: data = 32'hffff4a63;
        10'd244: data = 32'h000b59a9;
        10'd245: data = 32'hfffa7bf9;
        10'd246: data = 32'hfffa7b19;
        10'd247: data = 32'hfffd840a;
        10'd248: data = 32'hffd805c0;
        10'd249: data = 32'hffe1fad2;
        10'd250: data = 32'hfffecf1a;
        10'd251: data = 32'h005e2c61;
        10'd252: data = 32'h003ad167;
        10'd253: data = 32'h0028d8ff;
        10'd254: data = 32'hffde15f8;
        10'd255: data = 32'h00003fb9;
        10'd256: data = 32'h00350eeb;
        10'd257: data = 32'h003734c6;
        10'd258: data = 32'h000be25d;
        10'd259: data = 32'h000822d1;
        10'd260: data = 32'h001803fb;
        10'd261: data = 32'h00135952;
        10'd262: data = 32'h0012e97f;
        10'd263: data = 32'h000c5e77;
        10'd264: data = 32'h0001168e;
        10'd265: data = 32'h00158d56;
        10'd266: data = 32'h001f79bc;
        10'd267: data = 32'h0038c98a;
        10'd268: data = 32'h0002fd77;
        10'd269: data = 32'h00335982;
        10'd270: data = 32'h001a042f;
        10'd271: data = 32'h00028f81;
        10'd272: data = 32'hfffbc6d6;
        10'd273: data = 32'h001aa513;
        10'd274: data = 32'h001a3bcb;
        10'd275: data = 32'h00029050;
        10'd276: data = 32'hfffefeb3;
        10'd277: data = 32'h0015a132;
        10'd278: data = 32'h00106e84;
        10'd279: data = 32'h00171595;
        10'd280: data = 32'h00197a99;
        10'd281: data = 32'h000e127d;
        10'd282: data = 32'h0020b341;
        10'd283: data = 32'hfffefea2;
        10'd284: data = 32'hfffe21e8;
        10'd285: data = 32'h0004a884;
        10'd286: data = 32'h001759f0;
        10'd287: data = 32'hfffa86d4;
        10'd288: data = 32'hffff0341;
        10'd289: data = 32'h000e4389;
        10'd290: data = 32'hffc7cf35;
        10'd291: data = 32'hffc9472b;
        10'd292: data = 32'hffc39baf;
        10'd293: data = 32'hfffd003d;
        10'd294: data = 32'h0019667c;
        10'd295: data = 32'hffd9aefb;
        10'd296: data = 32'h00036d8a;
        10'd297: data = 32'h001d4a7b;
        10'd298: data = 32'hffc3db2f;
        10'd299: data = 32'hfff5055c;
        10'd300: data = 32'hfff2468c;
        10'd301: data = 32'hffdc1056;
        10'd302: data = 32'hffed0123;
        10'd303: data = 32'hffc828dd;
        10'd304: data = 32'h0001ed6f;
        10'd305: data = 32'hfff5a3d4;
        10'd306: data = 32'h0022a082;
        10'd307: data = 32'hffea4fce;
        10'd308: data = 32'h0006b673;
        10'd309: data = 32'hffd49823;
        10'd310: data = 32'hffe51b91;
        10'd311: data = 32'hffea3bc2;
        10'd312: data = 32'h001cad2e;
        10'd313: data = 32'h0014b841;
        10'd314: data = 32'hffe6a43f;
        10'd315: data = 32'h0001d9b4;
        10'd316: data = 32'h0002b880;
        10'd317: data = 32'h0001df1b;
        10'd318: data = 32'h001cafa6;
        10'd319: data = 32'hffff92a8;
        10'd320: data = 32'hffd75b5e;
        10'd321: data = 32'hffefe322;
        10'd322: data = 32'hfff07198;
        10'd323: data = 32'hffe42311;
        10'd324: data = 32'hffe5e484;
        10'd325: data = 32'hff8b49ab;
        10'd326: data = 32'hfffddbd2;
        10'd327: data = 32'hfffb09e7;
        10'd328: data = 32'h00004824;
        10'd329: data = 32'hfffd73cd;
        10'd330: data = 32'hffecc05a;
        10'd331: data = 32'hff9069e4;
        10'd332: data = 32'h0000502a;
        10'd333: data = 32'hfff84de2;
        10'd334: data = 32'hffefd700;
        10'd335: data = 32'hffdf91da;
        10'd336: data = 32'h0002ea2f;
        10'd337: data = 32'hfff13df8;
        10'd338: data = 32'hfffb1506;
        10'd339: data = 32'h0009f79f;
        10'd340: data = 32'h00020964;
        10'd341: data = 32'hfff3da7a;
        10'd342: data = 32'hfff9bc84;
        10'd343: data = 32'hfffae46c;
        10'd344: data = 32'hfff479f8;
        10'd345: data = 32'hfff71391;
        10'd346: data = 32'hffe8d774;
        10'd347: data = 32'h0002f170;
        10'd348: data = 32'h0001cfc7;
        10'd349: data = 32'h0000ef80;
        10'd350: data = 32'hfff8e692;
        10'd351: data = 32'h000281b7;
        10'd352: data = 32'h000dea9e;
        10'd353: data = 32'hfffeff83;
        10'd354: data = 32'h002464c4;
        10'd355: data = 32'h00138fe4;
        10'd356: data = 32'hffe961c7;
        10'd357: data = 32'hfff7bbbf;
        10'd358: data = 32'hffc2893a;
        10'd359: data = 32'hfff965b8;
        10'd360: data = 32'hffebeafd;
        10'd361: data = 32'hffc5536c;
        10'd362: data = 32'h00146970;
        10'd363: data = 32'h0002cac5;
        10'd364: data = 32'h000accba;
        10'd365: data = 32'h00090e64;
        10'd366: data = 32'h00163b42;
        10'd367: data = 32'h00161145;
        10'd368: data = 32'hffef94a4;
        10'd369: data = 32'h000c9157;
        10'd370: data = 32'hffb7b2e2;
        10'd371: data = 32'h001de123;
        10'd372: data = 32'hffed47f5;
        10'd373: data = 32'hfff7c10a;
        10'd374: data = 32'h001a1b64;
        10'd375: data = 32'h00044e2e;
        10'd376: data = 32'hffc3320a;
        10'd377: data = 32'hffc57650;
        10'd378: data = 32'h000be9c8;
        10'd379: data = 32'hfff2c18a;
        10'd380: data = 32'hffe91400;
        10'd381: data = 32'hfff1329a;
        10'd382: data = 32'hffc21ae7;
        10'd383: data = 32'hffdd4f8d;
        10'd384: data = 32'hfffa5957;
        10'd385: data = 32'h00000ca6;
        10'd386: data = 32'hfffd0456;
        10'd387: data = 32'hfffecbe1;
        10'd388: data = 32'hfffda4a6;
        10'd389: data = 32'h000efb88;
        10'd390: data = 32'h00044b79;
        10'd391: data = 32'hfffb74bf;
        10'd392: data = 32'h0015e870;
        10'd393: data = 32'hfff8379c;
        10'd394: data = 32'hfff5c34e;
        10'd395: data = 32'hffde07d2;
        10'd396: data = 32'hffed6c00;
        10'd397: data = 32'hfffa2c90;
        10'd398: data = 32'hfffe3b89;
        10'd399: data = 32'hfffa5d7a;
        10'd400: data = 32'h0019139d;
        10'd401: data = 32'hfffc79d3;
        10'd402: data = 32'hfff97e37;
        10'd403: data = 32'h000264c3;
        10'd404: data = 32'h0057987a;
        10'd405: data = 32'hfffef8ae;
        10'd406: data = 32'hfffd960b;
        10'd407: data = 32'hfffc32f7;
        10'd408: data = 32'hfff66a7c;
        10'd409: data = 32'hfff42404;
        10'd410: data = 32'hfff72cf9;
        10'd411: data = 32'h0047f302;
        10'd412: data = 32'hfffb9a41;
        10'd413: data = 32'h0059b954;
        10'd414: data = 32'hfff13f86;
        10'd415: data = 32'h0058aeb8;
        10'd416: data = 32'hffca5166;
        10'd417: data = 32'hfff94fd3;
        10'd418: data = 32'hfffd6af0;
        10'd419: data = 32'h000e9d1f;
        10'd420: data = 32'h0012f3b7;
        10'd421: data = 32'h00045c4e;
        10'd422: data = 32'h001ad097;
        10'd423: data = 32'h0039e50a;
        10'd424: data = 32'h00029711;
        10'd425: data = 32'h0021d803;
        10'd426: data = 32'h0000e4b8;
        10'd427: data = 32'hfffe6e2a;
        10'd428: data = 32'h00074069;
        10'd429: data = 32'h001748b4;
        10'd430: data = 32'h0005e458;
        10'd431: data = 32'h00062e32;
        10'd432: data = 32'hfffbeed0;
        10'd433: data = 32'hffe6ac6e;
        10'd434: data = 32'h001c4a1c;
        10'd435: data = 32'hffa12e40;
        10'd436: data = 32'h0001cffd;
        10'd437: data = 32'h001c31a2;
        10'd438: data = 32'hffc7c528;
        10'd439: data = 32'hfffe0c11;
        10'd440: data = 32'h001b3867;
        10'd441: data = 32'h00166d2c;
        10'd442: data = 32'h00057f2b;
        10'd443: data = 32'h0000d90d;
        10'd444: data = 32'hfffcfc08;
        10'd445: data = 32'hffff87a5;
        10'd446: data = 32'h0023499c;
        10'd447: data = 32'hfffd94ea;
        10'd448: data = 32'h0002ae99;
        10'd449: data = 32'h002dfb92;
        10'd450: data = 32'hffea929e;
        10'd451: data = 32'hffeec99a;
        10'd452: data = 32'hffe5ad2a;
        10'd453: data = 32'hfff93937;
        10'd454: data = 32'h000c4cd8;
        10'd455: data = 32'hffd15702;
        10'd456: data = 32'h000053ef;
        10'd457: data = 32'h0006c97d;
        10'd458: data = 32'hffe895ae;
        10'd459: data = 32'h00130a99;
        10'd460: data = 32'h0009f14a;
        10'd461: data = 32'hffd5fef4;
        10'd462: data = 32'hffbab17c;
        10'd463: data = 32'hffe1067e;
        10'd464: data = 32'h00001ac2;
        10'd465: data = 32'h00012c2e;
        10'd466: data = 32'h000d1a15;
        10'd467: data = 32'hfff69ccc;
        10'd468: data = 32'hfffead9b;
        10'd469: data = 32'hffcf4780;
        10'd470: data = 32'hfff3d3a3;
        10'd471: data = 32'hffdceae0;
        10'd472: data = 32'h000f17ea;
        10'd473: data = 32'h000c5c68;
        10'd474: data = 32'hffc571b8;
        10'd475: data = 32'hfffc9f48;
        10'd476: data = 32'h000387c9;
        10'd477: data = 32'hffff0259;
        10'd478: data = 32'h000f46d8;
        10'd479: data = 32'hfffff9fe;
        10'd480: data = 32'hfffb8c49;
        10'd481: data = 32'h0000a8d3;
        10'd482: data = 32'hfffac3f3;
        10'd483: data = 32'hfff909b6;
        10'd484: data = 32'hfffe978c;
        10'd485: data = 32'h00002003;
        10'd486: data = 32'hffe2ef0e;
        10'd487: data = 32'hfffd16db;
        10'd488: data = 32'h001e979d;
        10'd489: data = 32'hfff195ce;
        10'd490: data = 32'hfff6672d;
        10'd491: data = 32'h001932a4;
        10'd492: data = 32'h001262ea;
        10'd493: data = 32'hfffa013b;
        10'd494: data = 32'hfffd5faf;
        10'd495: data = 32'hfffd3288;
        10'd496: data = 32'h00329362;
        10'd497: data = 32'hfffbc697;
        10'd498: data = 32'hffd7ed46;
        10'd499: data = 32'h0001f2f5;
        10'd500: data = 32'h003fed84;
        10'd501: data = 32'hfffed26a;
        10'd502: data = 32'hfff8ce89;
        10'd503: data = 32'hfffc9bb3;
        10'd504: data = 32'hffe322a1;
        10'd505: data = 32'hffe45157;
        10'd506: data = 32'hfff8967c;
        10'd507: data = 32'hffbd158b;
        10'd508: data = 32'h0036d656;
        10'd509: data = 32'hfff56825;
        10'd510: data = 32'hffd93d0f;
        10'd511: data = 32'h00454478;
        default: data = 32'h00000000;
    endcase
end

endmodule

module layer2_bias_lut(
    input [3:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        4'd0: data = 32'hfffe45d8;
        4'd1: data = 32'hfff5a75b;
        4'd2: data = 32'h0001a33c;
        4'd3: data = 32'hfffe5477;
        4'd4: data = 32'hfff4a1c1;
        4'd5: data = 32'hffffe5db;
        4'd6: data = 32'hfffd2db7;
        4'd7: data = 32'hfffbac1e;
        4'd8: data = 32'hfffc63e4;
        4'd9: data = 32'h000c39e8;
        4'd10: data = 32'h000330bf;
        4'd11: data = 32'hfffb63be;
        4'd12: data = 32'hfff1a3c0;
        4'd13: data = 32'hfffb3d68;
        4'd14: data = 32'h000d3ea5;
        4'd15: data = 32'h00004450;
        default: data = 32'h00000000;
    endcase
end

endmodule

module output_weight_lut(
    input [5:0] addr,  // 4*16=64个权重
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        6'd0: data = 32'h006a38ff;
        6'd1: data = 32'hfff27759;
        6'd2: data = 32'h00751d7e;
        6'd3: data = 32'hffffa254;
        6'd4: data = 32'hfe926144;
        6'd5: data = 32'h0040f919;
        6'd6: data = 32'hfe678240;
        6'd7: data = 32'hfea9e37c;
        6'd8: data = 32'hffb0b45c;
        6'd9: data = 32'hfed8b91c;
        6'd10: data = 32'h000cbc3f;
        6'd11: data = 32'h005cf51d;
        6'd12: data = 32'hfe998634;
        6'd13: data = 32'hffc04d44;
        6'd14: data = 32'h0019000a;
        6'd15: data = 32'hfef422ee;
        6'd16: data = 32'hfff06539;
        6'd17: data = 32'h006194ca;
        6'd18: data = 32'hff6e0364;
        6'd19: data = 32'h0044c003;
        6'd20: data = 32'h002a82f2;
        6'd21: data = 32'hff3b46d7;
        6'd22: data = 32'hff48c1d0;
        6'd23: data = 32'h00164826;
        6'd24: data = 32'h003feeb7;
        6'd25: data = 32'hffad52e9;
        6'd26: data = 32'hff5f9d4a;
        6'd27: data = 32'hff81af22;
        6'd28: data = 32'h0028858a;
        6'd29: data = 32'h00156f89;
        6'd30: data = 32'hff870cd4;
        6'd31: data = 32'hfffa4397;
        6'd32: data = 32'h00535678;
        6'd33: data = 32'hff87863b;
        6'd34: data = 32'hffeaa47d;
        6'd35: data = 32'hffedba84;
        6'd36: data = 32'h00295d6c;
        6'd37: data = 32'h0022b250;
        6'd38: data = 32'h00ba4a2a;
        6'd39: data = 32'h001c60c4;
        6'd40: data = 32'hff7d5a17;
        6'd41: data = 32'hff524fff;
        6'd42: data = 32'h001f0538;
        6'd43: data = 32'h0043104c;
        6'd44: data = 32'h0025c39a;
        6'd45: data = 32'hff8593a0;
        6'd46: data = 32'h0045b244;
        6'd47: data = 32'hfffe4111;
        6'd48: data = 32'hff624b2f;
        6'd49: data = 32'h006af6d0;
        6'd50: data = 32'hffd2972c;
        6'd51: data = 32'hff942cec;
        6'd52: data = 32'h002817a0;
        6'd53: data = 32'hff2d4a7e;
        6'd54: data = 32'h00138c42;
        6'd55: data = 32'h000460c9;
        6'd56: data = 32'h0036f1da;
        6'd57: data = 32'h00e6f152;
        6'd58: data = 32'h006a8c67;
        6'd59: data = 32'hff91ecb2;
        6'd60: data = 32'h001b29d4;
        6'd61: data = 32'h00252730;
        6'd62: data = 32'h00762ab8;
        6'd63: data = 32'hffef827c;
        default: data = 32'h00000000;
    endcase
end

endmodule

module output_bias_lut(
    input [1:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        2'd0: data = 32'h002c9540;
        2'd1: data = 32'h004192ba;
        2'd2: data = 32'hffdfc185;
        2'd3: data = 32'hffbe5b27;
        default: data = 32'h00000000;
    endcase
end

endmodule

module bn1_scale_lut(
    input [4:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        5'd0: data = 32'h002703da;
        5'd1: data = 32'h0080eb7e;
        5'd2: data = 32'h00213d76;
        5'd3: data = 32'h0023779c;
        5'd4: data = 32'h00193868;
        5'd5: data = 32'h002e7b25;
        5'd6: data = 32'h014f8de8;
        5'd7: data = 32'h002dc6b3;
        5'd8: data = 32'h00bddf17;
        5'd9: data = 32'h0192d7a8;
        5'd10: data = 32'h001f218c;
        5'd11: data = 32'h004a2c5a;
        5'd12: data = 32'h0049574e;
        5'd13: data = 32'h002c10e3;
        5'd14: data = 32'h0024d4ef;
        5'd15: data = 32'h001d795f;
        5'd16: data = 32'h00aa3940;
        5'd17: data = 32'h003e8967;
        5'd18: data = 32'h01115418;
        5'd19: data = 32'h0048dc2a;
        5'd20: data = 32'h00c5b50c;
        5'd21: data = 32'h002004ce;
        5'd22: data = 32'h0052fed6;
        5'd23: data = 32'h003404be;
        5'd24: data = 32'h0157fbd8;
        5'd25: data = 32'h016c4d78;
        5'd26: data = 32'h001cbb9f;
        5'd27: data = 32'h00b0d698;
        5'd28: data = 32'h009b29b3;
        5'd29: data = 32'h00c7649e;
        5'd30: data = 32'h014a9b20;
        5'd31: data = 32'h00c888d5;
        default: data = 32'h00000000;
    endcase
end

endmodule

module bn1_shift_lut(
    input [4:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        5'd0: data = 32'h01043f2a;
        5'd1: data = 32'h01d9a1ec;
        5'd2: data = 32'hffe99376;
        5'd3: data = 32'hffc93a4b;
        5'd4: data = 32'hffcc7ff6;
        5'd5: data = 32'hfede4cca;
        5'd6: data = 32'hfffe76c5;
        5'd7: data = 32'h001d65ee;
        5'd8: data = 32'h00158d83;
        5'd9: data = 32'hff961270;
        5'd10: data = 32'hffe3172d;
        5'd11: data = 32'hfe4dd392;
        5'd12: data = 32'hff6c39d8;
        5'd13: data = 32'h004ccc0b;
        5'd14: data = 32'h005547e0;
        5'd15: data = 32'hffee8b60;
        5'd16: data = 32'h0011309b;
        5'd17: data = 32'h01129168;
        5'd18: data = 32'h0004258d;
        5'd19: data = 32'hfff8fa73;
        5'd20: data = 32'hffec7580;
        5'd21: data = 32'h0039abdb;
        5'd22: data = 32'hffd593a0;
        5'd23: data = 32'h00515e02;
        5'd24: data = 32'hffd240fd;
        5'd25: data = 32'hffd99b7f;
        5'd26: data = 32'h003fc84a;
        5'd27: data = 32'h0064b51e;
        5'd28: data = 32'hfff5170c;
        5'd29: data = 32'hffeddbe5;
        5'd30: data = 32'hffa1510e;
        5'd31: data = 32'h0018662a;
        default: data = 32'h00000000;
    endcase
end

endmodule

module bn2_scale_lut(
    input [3:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        4'd0: data = 32'h0037ac7a;
        4'd1: data = 32'h0071e8ca;
        4'd2: data = 32'h0037494b;
        4'd3: data = 32'h0060bd02;
        4'd4: data = 32'h0048b085;
        4'd5: data = 32'h0042c11c;
        4'd6: data = 32'h003169c4;
        4'd7: data = 32'h00593c4e;
        4'd8: data = 32'h0050a114;
        4'd9: data = 32'h003fa516;
        4'd10: data = 32'h0027c3b2;
        4'd11: data = 32'h0040efaa;
        4'd12: data = 32'h003f2624;
        4'd13: data = 32'h00766766;
        4'd14: data = 32'h0047ddc8;
        4'd15: data = 32'h0074da8f;
        default: data = 32'h00000000;
    endcase
end

endmodule

module bn2_shift_lut(
    input [3:0] addr,
    output reg [31:0] data
);

always @(*) begin
    case(addr)
        4'd0: data = 32'h0019c6f2;
        4'd1: data = 32'h0017b59f;
        4'd2: data = 32'h00165db6;
        4'd3: data = 32'h0029c034;
        4'd4: data = 32'h001f3912;
        4'd5: data = 32'h0023a174;
        4'd6: data = 32'h001c8838;
        4'd7: data = 32'hfff00504;
        4'd8: data = 32'hfed4aaee;
        4'd9: data = 32'h001c1019;
        4'd10: data = 32'h0061801d;
        4'd11: data = 32'h0035c843;
        4'd12: data = 32'hfff5084e;
        4'd13: data = 32'h0007a55e;
        4'd14: data = 32'h00154f6d;
        4'd15: data = 32'h0025997b;
        default: data = 32'h00000000;
    endcase
end

endmodule
