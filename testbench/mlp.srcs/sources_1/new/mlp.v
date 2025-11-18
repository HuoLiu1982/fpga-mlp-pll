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
        5'd0: data = 32'hfffffd97;
        5'd1: data = 32'h00000171;
        5'd2: data = 32'hffffea15;
        5'd3: data = 32'h00001633;
        5'd4: data = 32'h000005ea;
        5'd5: data = 32'h00000011;
        5'd6: data = 32'h00000380;
        5'd7: data = 32'hfffffb49;
        5'd8: data = 32'h000033d0;
        5'd9: data = 32'hfffffd19;
        5'd10: data = 32'hffffeff8;
        5'd11: data = 32'h00000ca6;
        5'd12: data = 32'h000000a0;
        5'd13: data = 32'h00000496;
        5'd14: data = 32'hfffffb12;
        5'd15: data = 32'hffffffa8;
        5'd16: data = 32'h00000501;
        5'd17: data = 32'h0000067b;
        5'd18: data = 32'h0000078d;
        5'd19: data = 32'hfffff739;
        5'd20: data = 32'h00000f54;
        5'd21: data = 32'hfffffe1e;
        5'd22: data = 32'hfffffaeb;
        5'd23: data = 32'hfffffc0b;
        5'd24: data = 32'hffffff59;
        5'd25: data = 32'hfffff737;
        5'd26: data = 32'h00000095;
        5'd27: data = 32'h000008cb;
        5'd28: data = 32'hffffffc7;
        5'd29: data = 32'hfffff124;
        5'd30: data = 32'h000001c9;
        5'd31: data = 32'h00000930;
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
        10'd0: data = 32'h00059932;
        10'd1: data = 32'h000477b1;
        10'd2: data = 32'h00101c09;
        10'd3: data = 32'h0003c6be;
        10'd4: data = 32'h00140e73;
        10'd5: data = 32'hffde7767;
        10'd6: data = 32'h000602d5;
        10'd7: data = 32'hfffad884;
        10'd8: data = 32'hfff373e8;
        10'd9: data = 32'h000fa2bf;
        10'd10: data = 32'hffd60f06;
        10'd11: data = 32'h001de2b7;
        10'd12: data = 32'hffe4fc72;
        10'd13: data = 32'hffb1cbe0;
        10'd14: data = 32'h000a1d9a;
        10'd15: data = 32'h00336c28;
        10'd16: data = 32'h001814fc;
        10'd17: data = 32'h00067d27;
        10'd18: data = 32'hffff81ea;
        10'd19: data = 32'hffececad;
        10'd20: data = 32'h001910b5;
        10'd21: data = 32'hffdfe8a2;
        10'd22: data = 32'hfff35d8c;
        10'd23: data = 32'h000a4de5;
        10'd24: data = 32'h002da511;
        10'd25: data = 32'hfff5bf8b;
        10'd26: data = 32'h00112df1;
        10'd27: data = 32'hfff58fbc;
        10'd28: data = 32'hffe37f4e;
        10'd29: data = 32'hffe89008;
        10'd30: data = 32'hffebb106;
        10'd31: data = 32'hfff3e2bc;
        10'd32: data = 32'hffff4e72;
        10'd33: data = 32'h00003aeb;
        10'd34: data = 32'h00066777;
        10'd35: data = 32'hfffba3bf;
        10'd36: data = 32'hfffec636;
        10'd37: data = 32'h0000f958;
        10'd38: data = 32'h00000ab4;
        10'd39: data = 32'h00384df4;
        10'd40: data = 32'hffdfeb34;
        10'd41: data = 32'hffffa090;
        10'd42: data = 32'h0004d0a3;
        10'd43: data = 32'hffff9791;
        10'd44: data = 32'hfffafa8c;
        10'd45: data = 32'hfffaf4a7;
        10'd46: data = 32'hffff4d6a;
        10'd47: data = 32'hfffece4e;
        10'd48: data = 32'h00156fb9;
        10'd49: data = 32'hfffc91bc;
        10'd50: data = 32'hfffeb93d;
        10'd51: data = 32'h00024930;
        10'd52: data = 32'h002f57a9;
        10'd53: data = 32'hfffe9c53;
        10'd54: data = 32'h003818d0;
        10'd55: data = 32'hfffe5620;
        10'd56: data = 32'hffff2d96;
        10'd57: data = 32'h004080d9;
        10'd58: data = 32'hfffd595c;
        10'd59: data = 32'h00358548;
        10'd60: data = 32'hfffa9349;
        10'd61: data = 32'hfff6777a;
        10'd62: data = 32'hfffa6114;
        10'd63: data = 32'h002df7d6;
        10'd64: data = 32'hffec6afc;
        10'd65: data = 32'hffe90bc5;
        10'd66: data = 32'h00059e10;
        10'd67: data = 32'hfffb39d8;
        10'd68: data = 32'hffe03484;
        10'd69: data = 32'hffde5018;
        10'd70: data = 32'h0009c41c;
        10'd71: data = 32'hfffc3fed;
        10'd72: data = 32'h00086406;
        10'd73: data = 32'hffd12490;
        10'd74: data = 32'h000e73f1;
        10'd75: data = 32'hffedd950;
        10'd76: data = 32'h00096972;
        10'd77: data = 32'h00559054;
        10'd78: data = 32'hffd230cc;
        10'd79: data = 32'hfff61ed5;
        10'd80: data = 32'hfff3391a;
        10'd81: data = 32'hfff00234;
        10'd82: data = 32'hffd35c9c;
        10'd83: data = 32'h000fda5d;
        10'd84: data = 32'hfff120b7;
        10'd85: data = 32'hffc85595;
        10'd86: data = 32'h0001f5e8;
        10'd87: data = 32'hffce7e59;
        10'd88: data = 32'hffda7f9a;
        10'd89: data = 32'h00013582;
        10'd90: data = 32'hffefa1b4;
        10'd91: data = 32'h0006e2b5;
        10'd92: data = 32'h0011639b;
        10'd93: data = 32'h00144117;
        10'd94: data = 32'h0015a31a;
        10'd95: data = 32'h0001f7e1;
        10'd96: data = 32'hffd64e03;
        10'd97: data = 32'hffca31a4;
        10'd98: data = 32'h00049f48;
        10'd99: data = 32'h000237e4;
        10'd100: data = 32'hffd10cf1;
        10'd101: data = 32'hfffc49f5;
        10'd102: data = 32'hffcde752;
        10'd103: data = 32'h0013672f;
        10'd104: data = 32'h000c3592;
        10'd105: data = 32'hfff9dc4f;
        10'd106: data = 32'h001a7cd7;
        10'd107: data = 32'hffd88593;
        10'd108: data = 32'h001adcae;
        10'd109: data = 32'hffe09c22;
        10'd110: data = 32'hffd5df67;
        10'd111: data = 32'hffedd39d;
        10'd112: data = 32'h0009c665;
        10'd113: data = 32'hffe644fe;
        10'd114: data = 32'hfffbd225;
        10'd115: data = 32'h0017e066;
        10'd116: data = 32'h000a5163;
        10'd117: data = 32'hffe00d56;
        10'd118: data = 32'h000ccd89;
        10'd119: data = 32'hffe3bc32;
        10'd120: data = 32'hffd61a85;
        10'd121: data = 32'h000de399;
        10'd122: data = 32'hffe1ec6f;
        10'd123: data = 32'h000d69b1;
        10'd124: data = 32'h00199bba;
        10'd125: data = 32'h00076d7d;
        10'd126: data = 32'h00163b60;
        10'd127: data = 32'h000bb872;
        10'd128: data = 32'hfffe332d;
        10'd129: data = 32'hffbff3e4;
        10'd130: data = 32'h00087bd9;
        10'd131: data = 32'h000bf04a;
        10'd132: data = 32'h000ccd57;
        10'd133: data = 32'h0017443d;
        10'd134: data = 32'h00076991;
        10'd135: data = 32'hfffc8106;
        10'd136: data = 32'hfff9d559;
        10'd137: data = 32'hff934918;
        10'd138: data = 32'hfff45612;
        10'd139: data = 32'h000eb59f;
        10'd140: data = 32'hffee563e;
        10'd141: data = 32'hfff49c67;
        10'd142: data = 32'h000c356e;
        10'd143: data = 32'hffef257b;
        10'd144: data = 32'h00114b80;
        10'd145: data = 32'h0016250a;
        10'd146: data = 32'h0006e464;
        10'd147: data = 32'hffee579e;
        10'd148: data = 32'h001af93e;
        10'd149: data = 32'h000c6d99;
        10'd150: data = 32'hfff816e9;
        10'd151: data = 32'hfffddf98;
        10'd152: data = 32'h0004ae2a;
        10'd153: data = 32'h0005dee9;
        10'd154: data = 32'hfff4575f;
        10'd155: data = 32'h00002d13;
        10'd156: data = 32'hfffb42ed;
        10'd157: data = 32'hfff5e687;
        10'd158: data = 32'hffe250ad;
        10'd159: data = 32'hfffdc7a8;
        10'd160: data = 32'hfffd6829;
        10'd161: data = 32'h0000046e;
        10'd162: data = 32'h0008e39a;
        10'd163: data = 32'h00192ff4;
        10'd164: data = 32'h00004399;
        10'd165: data = 32'hfffb851c;
        10'd166: data = 32'hfffe4685;
        10'd167: data = 32'h00177ca6;
        10'd168: data = 32'h002e0f0a;
        10'd169: data = 32'h0001021f;
        10'd170: data = 32'hfff81b0b;
        10'd171: data = 32'hfffd827b;
        10'd172: data = 32'h0000baad;
        10'd173: data = 32'hffef5bf9;
        10'd174: data = 32'h0000a6f7;
        10'd175: data = 32'hfffce63e;
        10'd176: data = 32'h002fbad0;
        10'd177: data = 32'hfffcb3b0;
        10'd178: data = 32'hfffee3c5;
        10'd179: data = 32'hfffe2b55;
        10'd180: data = 32'h003bb6f7;
        10'd181: data = 32'hfffef2ac;
        10'd182: data = 32'h001f060e;
        10'd183: data = 32'hffffb932;
        10'd184: data = 32'hfffe7f54;
        10'd185: data = 32'h00403f0a;
        10'd186: data = 32'hfffdb4d4;
        10'd187: data = 32'h0013ac88;
        10'd188: data = 32'hfffee141;
        10'd189: data = 32'h00275790;
        10'd190: data = 32'h0000f02b;
        10'd191: data = 32'hffe89987;
        10'd192: data = 32'hfffe439b;
        10'd193: data = 32'h0000123d;
        10'd194: data = 32'h001af2a4;
        10'd195: data = 32'h00351845;
        10'd196: data = 32'hffff5dbd;
        10'd197: data = 32'hfff80a02;
        10'd198: data = 32'hfffe30bb;
        10'd199: data = 32'h002b4dd1;
        10'd200: data = 32'h00351705;
        10'd201: data = 32'h000036bb;
        10'd202: data = 32'h0005e672;
        10'd203: data = 32'hffff1fbc;
        10'd204: data = 32'hffe55f39;
        10'd205: data = 32'hffe3cc4b;
        10'd206: data = 32'hfffda63e;
        10'd207: data = 32'hfffce714;
        10'd208: data = 32'h002b05db;
        10'd209: data = 32'hfffe2cc9;
        10'd210: data = 32'hfffe665c;
        10'd211: data = 32'hffea619a;
        10'd212: data = 32'h002f1508;
        10'd213: data = 32'hfffe80b9;
        10'd214: data = 32'h0011d5f2;
        10'd215: data = 32'hffff6da0;
        10'd216: data = 32'hffff8cc8;
        10'd217: data = 32'h0006fa16;
        10'd218: data = 32'hfffe17b7;
        10'd219: data = 32'h002bee2f;
        10'd220: data = 32'hffff9098;
        10'd221: data = 32'h0027448c;
        10'd222: data = 32'hfff856f1;
        10'd223: data = 32'h004c3b33;
        10'd224: data = 32'hffef0348;
        10'd225: data = 32'hfff0cb9f;
        10'd226: data = 32'h0010f5e0;
        10'd227: data = 32'h000f6d87;
        10'd228: data = 32'hffe0eb65;
        10'd229: data = 32'h000ce571;
        10'd230: data = 32'hffe99003;
        10'd231: data = 32'hfff82e91;
        10'd232: data = 32'hfff15ac6;
        10'd233: data = 32'h00025752;
        10'd234: data = 32'h00352ad5;
        10'd235: data = 32'hffe48282;
        10'd236: data = 32'h001a9f35;
        10'd237: data = 32'h000436d7;
        10'd238: data = 32'hffd9f9d9;
        10'd239: data = 32'hffe92790;
        10'd240: data = 32'hfff757cc;
        10'd241: data = 32'hffde4292;
        10'd242: data = 32'hfff942d4;
        10'd243: data = 32'h001f8e1a;
        10'd244: data = 32'hffed4e51;
        10'd245: data = 32'hffd78dc5;
        10'd246: data = 32'hfff05627;
        10'd247: data = 32'hffdfcc63;
        10'd248: data = 32'hffe4fc60;
        10'd249: data = 32'hfff6b1e3;
        10'd250: data = 32'hffd88445;
        10'd251: data = 32'hffee793e;
        10'd252: data = 32'h001dc42c;
        10'd253: data = 32'hfff73065;
        10'd254: data = 32'h001546e6;
        10'd255: data = 32'hfff4f29c;
        10'd256: data = 32'hffeef023;
        10'd257: data = 32'hfffcbdce;
        10'd258: data = 32'h00044aa6;
        10'd259: data = 32'h0004ad2b;
        10'd260: data = 32'hfff891d7;
        10'd261: data = 32'hffcc1b89;
        10'd262: data = 32'hfffcca0e;
        10'd263: data = 32'h0007f727;
        10'd264: data = 32'h00112bf0;
        10'd265: data = 32'h00014649;
        10'd266: data = 32'hfffb7b36;
        10'd267: data = 32'hfff30930;
        10'd268: data = 32'hfff9b2a4;
        10'd269: data = 32'h00391ee9;
        10'd270: data = 32'hfff0a962;
        10'd271: data = 32'hffcf2659;
        10'd272: data = 32'hfffb1dd0;
        10'd273: data = 32'hfff13e81;
        10'd274: data = 32'hff948cc8;
        10'd275: data = 32'h0000a3d6;
        10'd276: data = 32'h0003ff89;
        10'd277: data = 32'hffb234e3;
        10'd278: data = 32'h001e63de;
        10'd279: data = 32'hffc6939d;
        10'd280: data = 32'hfff56235;
        10'd281: data = 32'h0002ef1b;
        10'd282: data = 32'hfff7622e;
        10'd283: data = 32'h0012b2c4;
        10'd284: data = 32'hffff1b08;
        10'd285: data = 32'h00231d84;
        10'd286: data = 32'h000c69d0;
        10'd287: data = 32'h0010ed48;
        10'd288: data = 32'h00021e03;
        10'd289: data = 32'hfff35eb9;
        10'd290: data = 32'h00103806;
        10'd291: data = 32'h0011342b;
        10'd292: data = 32'hffed7b3b;
        10'd293: data = 32'h00135412;
        10'd294: data = 32'hffd627e7;
        10'd295: data = 32'hffee52ef;
        10'd296: data = 32'hfff4d044;
        10'd297: data = 32'h00092610;
        10'd298: data = 32'hfff15f82;
        10'd299: data = 32'hffe3a4a0;
        10'd300: data = 32'hfff5f114;
        10'd301: data = 32'h0023088e;
        10'd302: data = 32'hffda3810;
        10'd303: data = 32'hffca55aa;
        10'd304: data = 32'hfffa0ef6;
        10'd305: data = 32'hffca1abd;
        10'd306: data = 32'h000bc05a;
        10'd307: data = 32'hfffb45b1;
        10'd308: data = 32'hfff68215;
        10'd309: data = 32'hffd46bd6;
        10'd310: data = 32'hfff611b3;
        10'd311: data = 32'hffedaee8;
        10'd312: data = 32'hfff15d4c;
        10'd313: data = 32'hfff14bea;
        10'd314: data = 32'hffc96fd3;
        10'd315: data = 32'hfff224d1;
        10'd316: data = 32'hffe2b78b;
        10'd317: data = 32'hfff1e32a;
        10'd318: data = 32'hffe04c70;
        10'd319: data = 32'hfff51866;
        10'd320: data = 32'hfff4b206;
        10'd321: data = 32'hfffaf870;
        10'd322: data = 32'h002794dc;
        10'd323: data = 32'h001e357b;
        10'd324: data = 32'hfff8f568;
        10'd325: data = 32'h00195576;
        10'd326: data = 32'hfff422f3;
        10'd327: data = 32'hfff360cf;
        10'd328: data = 32'hfff2701e;
        10'd329: data = 32'h000c41b5;
        10'd330: data = 32'hffdb729c;
        10'd331: data = 32'hfff70a65;
        10'd332: data = 32'hffd5e75e;
        10'd333: data = 32'hfff3735c;
        10'd334: data = 32'hfff21f20;
        10'd335: data = 32'hffed6f42;
        10'd336: data = 32'h00024470;
        10'd337: data = 32'hfff0de93;
        10'd338: data = 32'hfffb7aa8;
        10'd339: data = 32'hffe28bad;
        10'd340: data = 32'h00025ce4;
        10'd341: data = 32'hffee672d;
        10'd342: data = 32'hfff93b1c;
        10'd343: data = 32'hfff44f97;
        10'd344: data = 32'hfffac2e2;
        10'd345: data = 32'hfffea6a8;
        10'd346: data = 32'hffef82ae;
        10'd347: data = 32'hfff6d4aa;
        10'd348: data = 32'hffc7e10f;
        10'd349: data = 32'hffebd1d7;
        10'd350: data = 32'hffc9291f;
        10'd351: data = 32'hfff797f6;
        10'd352: data = 32'hffbdc05a;
        10'd353: data = 32'hffc265b6;
        10'd354: data = 32'hffd0d1b3;
        10'd355: data = 32'h000763c4;
        10'd356: data = 32'hffe3dc9c;
        10'd357: data = 32'hffeb08d3;
        10'd358: data = 32'hffe2e65f;
        10'd359: data = 32'h000aa1ca;
        10'd360: data = 32'h0010b86a;
        10'd361: data = 32'hffe4ae15;
        10'd362: data = 32'h001c3ed8;
        10'd363: data = 32'hffe50c9e;
        10'd364: data = 32'h0026dbe0;
        10'd365: data = 32'hffca68ca;
        10'd366: data = 32'hffeb1388;
        10'd367: data = 32'hfff2d21c;
        10'd368: data = 32'h00085fed;
        10'd369: data = 32'hfffe1fbd;
        10'd370: data = 32'hfff4fd61;
        10'd371: data = 32'h002b6e2c;
        10'd372: data = 32'h000645b5;
        10'd373: data = 32'hffeb60fa;
        10'd374: data = 32'h000b27e6;
        10'd375: data = 32'hffe93530;
        10'd376: data = 32'hffd7ee84;
        10'd377: data = 32'h0002a94a;
        10'd378: data = 32'hfff7d360;
        10'd379: data = 32'h000d266c;
        10'd380: data = 32'h00211843;
        10'd381: data = 32'h0016f449;
        10'd382: data = 32'h000d16aa;
        10'd383: data = 32'h000b91e1;
        10'd384: data = 32'h0019b9cc;
        10'd385: data = 32'h000b944c;
        10'd386: data = 32'hffe4110f;
        10'd387: data = 32'hffb3e11b;
        10'd388: data = 32'h001c859e;
        10'd389: data = 32'h000751e4;
        10'd390: data = 32'h001d6817;
        10'd391: data = 32'hffea2950;
        10'd392: data = 32'hffe4fa14;
        10'd393: data = 32'h00145ca4;
        10'd394: data = 32'h00014950;
        10'd395: data = 32'h0013d9c8;
        10'd396: data = 32'hffdde3cb;
        10'd397: data = 32'hffd595ba;
        10'd398: data = 32'h00155c84;
        10'd399: data = 32'h00102fd7;
        10'd400: data = 32'hffe06c37;
        10'd401: data = 32'h00174e15;
        10'd402: data = 32'h001ad5c5;
        10'd403: data = 32'hfff6f404;
        10'd404: data = 32'hffe9b66a;
        10'd405: data = 32'h00095cc9;
        10'd406: data = 32'hffe14f90;
        10'd407: data = 32'h000456d0;
        10'd408: data = 32'h000b890f;
        10'd409: data = 32'hffe14d54;
        10'd410: data = 32'h00187946;
        10'd411: data = 32'hffe1cd67;
        10'd412: data = 32'h0008d637;
        10'd413: data = 32'hffdd7470;
        10'd414: data = 32'h0000a4ac;
        10'd415: data = 32'hffe741a0;
        10'd416: data = 32'h000beb6f;
        10'd417: data = 32'h0008f2f7;
        10'd418: data = 32'h0024a0c4;
        10'd419: data = 32'h0019d9f2;
        10'd420: data = 32'h00112f76;
        10'd421: data = 32'h0024a570;
        10'd422: data = 32'h0040d0df;
        10'd423: data = 32'h00097d25;
        10'd424: data = 32'h0005a591;
        10'd425: data = 32'h0018d6b2;
        10'd426: data = 32'hfff53c26;
        10'd427: data = 32'h0014cc01;
        10'd428: data = 32'h000726e1;
        10'd429: data = 32'h001b5b56;
        10'd430: data = 32'h00113f25;
        10'd431: data = 32'h0025d299;
        10'd432: data = 32'h0006029b;
        10'd433: data = 32'h001c95b4;
        10'd434: data = 32'h0008a5a7;
        10'd435: data = 32'h0003f8b0;
        10'd436: data = 32'h00046419;
        10'd437: data = 32'h000f2ad7;
        10'd438: data = 32'h000a6b62;
        10'd439: data = 32'h000bdd31;
        10'd440: data = 32'h0014e00a;
        10'd441: data = 32'h0003fa9d;
        10'd442: data = 32'h00297268;
        10'd443: data = 32'h00050a7c;
        10'd444: data = 32'h0000d159;
        10'd445: data = 32'h000b1516;
        10'd446: data = 32'h000c13e2;
        10'd447: data = 32'h0008ca44;
        10'd448: data = 32'hffeb7f42;
        10'd449: data = 32'hffe76d16;
        10'd450: data = 32'h001237f2;
        10'd451: data = 32'hfff8b539;
        10'd452: data = 32'hffdf9051;
        10'd453: data = 32'hffe4fbf3;
        10'd454: data = 32'hfff569d4;
        10'd455: data = 32'h00041dd1;
        10'd456: data = 32'h000cf7cf;
        10'd457: data = 32'hffdcf354;
        10'd458: data = 32'hfff9bd28;
        10'd459: data = 32'hffeac700;
        10'd460: data = 32'h000defd4;
        10'd461: data = 32'h00656674;
        10'd462: data = 32'hffd72d95;
        10'd463: data = 32'hfffaa6e8;
        10'd464: data = 32'hffe8d162;
        10'd465: data = 32'hffdff466;
        10'd466: data = 32'hffecab64;
        10'd467: data = 32'h000a3c15;
        10'd468: data = 32'hfff54df5;
        10'd469: data = 32'hffe25f8b;
        10'd470: data = 32'h000241e3;
        10'd471: data = 32'hffe0ea5a;
        10'd472: data = 32'hffca2991;
        10'd473: data = 32'h000ba94e;
        10'd474: data = 32'hffd7c988;
        10'd475: data = 32'h0002b6ea;
        10'd476: data = 32'h00098578;
        10'd477: data = 32'h0012ee56;
        10'd478: data = 32'h0021dd2e;
        10'd479: data = 32'h000dee70;
        10'd480: data = 32'h002130ef;
        10'd481: data = 32'h0016fabf;
        10'd482: data = 32'h00033c6d;
        10'd483: data = 32'h0005613c;
        10'd484: data = 32'h000a0aac;
        10'd485: data = 32'h000b2eea;
        10'd486: data = 32'h00178d10;
        10'd487: data = 32'hffd981aa;
        10'd488: data = 32'hffceaf98;
        10'd489: data = 32'h0007a0de;
        10'd490: data = 32'hfffb2784;
        10'd491: data = 32'h000eb9a8;
        10'd492: data = 32'hffd02940;
        10'd493: data = 32'hfff8d5d9;
        10'd494: data = 32'h00100f60;
        10'd495: data = 32'h000f85a1;
        10'd496: data = 32'hffd86742;
        10'd497: data = 32'h000b9ad4;
        10'd498: data = 32'h000977f9;
        10'd499: data = 32'hffc5551c;
        10'd500: data = 32'hffdc38da;
        10'd501: data = 32'h001fd15e;
        10'd502: data = 32'hffdc4a86;
        10'd503: data = 32'h0009b2ef;
        10'd504: data = 32'h0012128a;
        10'd505: data = 32'hffdfffb8;
        10'd506: data = 32'h00148ad6;
        10'd507: data = 32'hffce5da1;
        10'd508: data = 32'hfffd66ad;
        10'd509: data = 32'hffcdafac;
        10'd510: data = 32'h0004ba28;
        10'd511: data = 32'hffd68bb1;
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
        4'd0: data = 32'h00000057;
        4'd1: data = 32'hfffff003;
        4'd2: data = 32'hfffffc9f;
        4'd3: data = 32'h00000074;
        4'd4: data = 32'hfffffd33;
        4'd5: data = 32'h00000170;
        4'd6: data = 32'hffffffe3;
        4'd7: data = 32'hfffffca0;
        4'd8: data = 32'h000000cc;
        4'd9: data = 32'h000004e0;
        4'd10: data = 32'h0000035f;
        4'd11: data = 32'h000003e9;
        4'd12: data = 32'hfffffab2;
        4'd13: data = 32'h0000016b;
        4'd14: data = 32'hfffffbf0;
        4'd15: data = 32'h000000ba;
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
        6'd0: data = 32'h0024a39a;
        6'd1: data = 32'hfea37bf4;
        6'd2: data = 32'hffe63cfc;
        6'd3: data = 32'hff7fb7a3;
        6'd4: data = 32'h00400d0d;
        6'd5: data = 32'hfe96dce6;
        6'd6: data = 32'hfe7c251c;
        6'd7: data = 32'hffa3ea12;
        6'd8: data = 32'hff8b03cb;
        6'd9: data = 32'hff362206;
        6'd10: data = 32'hff1ea1b1;
        6'd11: data = 32'hffa55aa5;
        6'd12: data = 32'h0048d722;
        6'd13: data = 32'h000fc915;
        6'd14: data = 32'hffa596c8;
        6'd15: data = 32'h0022c0e6;
        6'd16: data = 32'h000775f7;
        6'd17: data = 32'h00307b0f;
        6'd18: data = 32'hffae06b8;
        6'd19: data = 32'hff6d9cb2;
        6'd20: data = 32'h00412067;
        6'd21: data = 32'h002d49f8;
        6'd22: data = 32'h000d9eb3;
        6'd23: data = 32'hff8d13b3;
        6'd24: data = 32'h00362df2;
        6'd25: data = 32'hffd756ca;
        6'd26: data = 32'h002fe1bc;
        6'd27: data = 32'hffcf7227;
        6'd28: data = 32'hffcc613f;
        6'd29: data = 32'h007f65fc;
        6'd30: data = 32'hffcd7be6;
        6'd31: data = 32'hff740cef;
        6'd32: data = 32'h00020349;
        6'd33: data = 32'h0028f063;
        6'd34: data = 32'hff9e1972;
        6'd35: data = 32'h003a7a51;
        6'd36: data = 32'hffc7525f;
        6'd37: data = 32'h00264ebe;
        6'd38: data = 32'h0004e238;
        6'd39: data = 32'h003610a0;
        6'd40: data = 32'hff4153a0;
        6'd41: data = 32'h00b46ea3;
        6'd42: data = 32'h002f6434;
        6'd43: data = 32'h00405324;
        6'd44: data = 32'h00329882;
        6'd45: data = 32'hffe8e1af;
        6'd46: data = 32'hffea21c5;
        6'd47: data = 32'h000368ac;
        6'd48: data = 32'hff964a40;
        6'd49: data = 32'h002603b7;
        6'd50: data = 32'h0086d2b4;
        6'd51: data = 32'h00383cd8;
        6'd52: data = 32'hff899fb0;
        6'd53: data = 32'h00212eb4;
        6'd54: data = 32'h00003b46;
        6'd55: data = 32'h008a10e8;
        6'd56: data = 32'h0049478a;
        6'd57: data = 32'hff6c4251;
        6'd58: data = 32'hffc1eebc;
        6'd59: data = 32'h002e0d14;
        6'd60: data = 32'hffe64bc0;
        6'd61: data = 32'hffe33dfd;
        6'd62: data = 32'h005df556;
        6'd63: data = 32'hffb0aaee;
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
        2'd0: data = 32'h0043ba9a;
        2'd1: data = 32'hfffc8150;
        2'd2: data = 32'hffedf2db;
        2'd3: data = 32'hffdd3701;
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
        5'd0: data = 32'h00741310;
        5'd1: data = 32'h008259be;
        5'd2: data = 32'h009369c1;
        5'd3: data = 32'h00acc16d;
        5'd4: data = 32'h006745a2;
        5'd5: data = 32'h0061e5a6;
        5'd6: data = 32'h0049b26a;
        5'd7: data = 32'h015db826;
        5'd8: data = 32'h0181342e;
        5'd9: data = 32'h007a97a5;
        5'd10: data = 32'h01787b6c;
        5'd11: data = 32'h00777feb;
        5'd12: data = 32'h01769adc;
        5'd13: data = 32'h0084c3fa;
        5'd14: data = 32'h004bab56;
        5'd15: data = 32'h00672424;
        5'd16: data = 32'h022aede4;
        5'd17: data = 32'h006c95b4;
        5'd18: data = 32'h008fbb5d;
        5'd19: data = 32'h014ad4f0;
        5'd20: data = 32'h0188cd9a;
        5'd21: data = 32'h00555e19;
        5'd22: data = 32'h017a03e4;
        5'd23: data = 32'h0060a72f;
        5'd24: data = 32'h007d51fe;
        5'd25: data = 32'h01449f5e;
        5'd26: data = 32'h00560c12;
        5'd27: data = 32'h019f6d96;
        5'd28: data = 32'h01723286;
        5'd29: data = 32'h016aded4;
        5'd30: data = 32'h0197bcea;
        5'd31: data = 32'h01860cb2;
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
        5'd0: data = 32'h012e131c;
        5'd1: data = 32'hffe6e732;
        5'd2: data = 32'hfdb451f8;
        5'd3: data = 32'hfe13f076;
        5'd4: data = 32'h004cf0e8;
        5'd5: data = 32'hfee898fc;
        5'd6: data = 32'h00810967;
        5'd7: data = 32'h001dd034;
        5'd8: data = 32'h002f12cd;
        5'd9: data = 32'hfec2aacc;
        5'd10: data = 32'h00f491f5;
        5'd11: data = 32'h0099072a;
        5'd12: data = 32'h008914e2;
        5'd13: data = 32'h03215338;
        5'd14: data = 32'h0076c0b6;
        5'd15: data = 32'h006a8830;
        5'd16: data = 32'hffff58ca;
        5'd17: data = 32'h00781e98;
        5'd18: data = 32'hffacc1d2;
        5'd19: data = 32'h0055e6a9;
        5'd20: data = 32'hfffe9691;
        5'd21: data = 32'h007593ce;
        5'd22: data = 32'h001d04d6;
        5'd23: data = 32'h0050732c;
        5'd24: data = 32'h0019f791;
        5'd25: data = 32'h000e8daf;
        5'd26: data = 32'h006afdb5;
        5'd27: data = 32'h0025a982;
        5'd28: data = 32'h01557b08;
        5'd29: data = 32'h00139926;
        5'd30: data = 32'h010f249c;
        5'd31: data = 32'h0065982a;
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
        4'd0: data = 32'h0037a87f;
        4'd1: data = 32'h0040d433;
        4'd2: data = 32'h00268ae5;
        4'd3: data = 32'h002dad81;
        4'd4: data = 32'h001d799c;
        4'd5: data = 32'h003ed2cb;
        4'd6: data = 32'h002dc802;
        4'd7: data = 32'h0031275a;
        4'd8: data = 32'h00202b7f;
        4'd9: data = 32'h0026e947;
        4'd10: data = 32'h0033bd09;
        4'd11: data = 32'h0033223e;
        4'd12: data = 32'h003670d3;
        4'd13: data = 32'h002a5d7e;
        4'd14: data = 32'h00310496;
        4'd15: data = 32'h00319851;
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
        4'd0: data = 32'h0011175c;
        4'd1: data = 32'hfff76bb6;
        4'd2: data = 32'h000c454d;
        4'd3: data = 32'hfffb484a;
        4'd4: data = 32'h00038003;
        4'd5: data = 32'hfffb0cf6;
        4'd6: data = 32'hfffa552e;
        4'd7: data = 32'h00056099;
        4'd8: data = 32'h00030792;
        4'd9: data = 32'h0007f6c0;
        4'd10: data = 32'h000b85d4;
        4'd11: data = 32'h00069e3e;
        4'd12: data = 32'h000de741;
        4'd13: data = 32'hff457428;
        4'd14: data = 32'h00131a32;
        4'd15: data = 32'h000ed2fa;
        default: data = 32'h00000000;
    endcase
end

endmodule
