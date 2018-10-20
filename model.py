# full assembly of the sub-parts to form the complete net

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.between1 = bet_model()
        self.between2 = bet_model()
        self.between3 = bet_model()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6, x7, x8 = self.between1(x5, x4, x3)
        x6, x7, x8 = self.between2(x6, x7, x8)
        x6, x7, x8 = self.between3(x6, x7, x8)
        x = self.up1(x6, x7)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class up_bet(nn.Module):
    def __init__(self):
        super(up_bet, self).__init__()
        self.up1 = up_between(512, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)

    def forward(self, x_low, x_mid, x_top):
        x_low_out = self.up1(x_low)
        x_mid_out = self.up2(x_mid, x_low_out)
        x_top_out = self.up3(x_top, x_mid_out)
        return x_low_out, x_mid_out, x_top_out

class down_bet(nn.Module):
    def __init__(self):
        super(down_bet, self).__init__()
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x_low, x_mid, x_top):
        x_top_out = self.down2(x_top)
        x_temp = torch.cat([x_top_out, x_mid], dim=1)
        x_mid_out = self.down3(x_temp)
        x_temp = torch.cat([x_mid_out, x_low])
        x_low_out = self.down4(x_temp)
        return x_low_out, x_mid_out, x_top_out

#model = nn.Sequential([up_bet(), down_bet()])


class bet_model(nn.Module):
    def __init__(self):
        self.up = up_between()
        self.down = down_between()

    def forward(self, x_low, x_mid, x_top):
        x_low, x_mid, x_top = self.up(x_low, x_mid, x_top)
        x_low, x_mid, x_top = self.down(x_low, x_mid, x_top)
        return x_low, x_mid, x_top



if __name__=="__main__":
    model = UNet(3, 3)
    x = torch.randn(1,3,224,224)
    y = model(x)
    print(y.size())
