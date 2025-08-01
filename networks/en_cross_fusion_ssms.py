import torch
import torch.nn as nn
from networks.A_ConvModule import BasicResBlock, DownsamolingBlock, Upsample_Layer_nearest, DecoderResBlock
from networks.ssm_mamba import L2DC
from networks.new_ib import ODL

class SGODL(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(SGODL, self).__init__()

        self.conv6 = nn.Sequential(nn.Conv3d(16, out_channel, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(out_channel),nn.ReLU(inplace=True))

        # encoder1
        self.encoder11 = BasicResBlock(in_channel, 16)
        self.Down11 = DownsamolingBlock(16, 32)
        self.encoder12 = BasicResBlock(32, 32)
        self.Down12 = DownsamolingBlock(32, 64)
        self.encoder13 = BasicResBlock(64, 64)
        self.Down13 = DownsamolingBlock(64, 128)
        self.encoder14 = BasicResBlock(128, 128)
        self.Down14 = DownsamolingBlock(128, 256)
        self.encoder15 = BasicResBlock(256, 256)
        # encoder2
        self.encoder21 = BasicResBlock(in_channel, 16)
        self.Down21 = DownsamolingBlock(16, 32)
        self.encoder22 = BasicResBlock(32, 32)
        self.Down22 = DownsamolingBlock(32, 64)
        self.encoder23 = BasicResBlock(64, 64)
        self.Down23 = DownsamolingBlock(64, 128)
        self.encoder24 = BasicResBlock(128, 128)
        self.Down24 = DownsamolingBlock(128, 256)
        self.encoder25 = BasicResBlock(256, 256)
        # decoder
        self.upsample14 = Upsample_Layer_nearest(256, 128)
        self.decoder11 = DecoderResBlock(256, 128)
        self.upsample13 = Upsample_Layer_nearest(128, 64)
        self.decoder12 = DecoderResBlock(128, 64)
        self.upsample12 = Upsample_Layer_nearest(64, 32)
        self.decoder13 = DecoderResBlock(64, 32)
        self.upsample11 = Upsample_Layer_nearest(32, 16)
        self.decoder14 = DecoderResBlock(32, 16)

        self.ib1 = ODL(16, 16)
        self.ib2 = ODL(32, 32)
        self.ib3 = ODL(64, 64)
        self.ib4 = ODL(128, 128)

        self.ssm = L2DC(channels=256)

    def forward(self, data):
        if data.size(0) == 1:
            in1 = data
            in2 = data
        else:
            in1 = data[:2]
            in2 = data[2:]

        # Encoder 1
        out1 = self.encoder11(in2)
        out11 = out1
        out1 = self.Down11(out1)

        out1 = self.encoder12(out1)
        out12 = out1
        out1 = self.Down12(out1)

        out1 = self.encoder13(out1)
        out13 = out1
        out1 = self.Down13(out1)

        out1 = self.encoder14(out1)
        out14 = out1
        out1 = self.Down14(out1)

        out1 = self.encoder15(out1)

        # Encoder 2
        out2 = self.encoder21(in1)
        out21 = out2
        out2 = self.ib1(out2, out11)
        out2 = self.Down21(out2)

        out2 = self.encoder22(out2)
        out22 = out2
        out2  = self.ib2(out2, out12)
        out2 = self.Down22(out2)

        out2 = self.encoder23(out2)
        out23 = out2
        out2  = self.ib3(out2, out13)
        out2 = self.Down23(out2)

        out2 = self.encoder24(out2)
        out24 = out2
        out2  = self.ib4(out2, out14)
        out2 = self.Down24(out2)

        out2 = self.encoder25(out2)

        state_out = self.ssm(out2, out1)

        # Decoder
        out1 = self.upsample14(state_out)
        skip14 = torch.cat((out1, out24), dim=1)
        out1 = self.decoder11(skip14)

        out1 = self.upsample13(out1)
        skip13 = torch.cat((out1, out23), dim=1)
        out1 = self.decoder12(skip13)

        out1 = self.upsample12(out1)
        skip12 = torch.cat((out1, out22), dim=1)
        out1 = self.decoder13(skip12)

        out1 = self.upsample11(out1)
        skip11 = torch.cat((out1, out21), dim=1)
        out1 = self.decoder14(skip11)

        out_ = self.conv6(out1)

        return out_





