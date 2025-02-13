import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, conv2_layer_count=8):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            conv2_layer_count -- origin is 8, residual block+self attention is 11
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, layer=1, conv2_layer_count=conv2_layer_count)
        for index in range(num_downs - 5):  # add intermediate layers with ngf * 8 filtersv
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=index+2, use_dropout=use_dropout, conv2_layer_count=conv2_layer_count)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=5, conv2_layer_count=conv2_layer_count)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=6, conv2_layer_count=conv2_layer_count)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=7, conv2_layer_count=conv2_layer_count)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, layer=8, conv2_layer_count=conv2_layer_count)

    def forward(self, input, style=None):
        return self.model(input, style)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, norm_layer=nn.BatchNorm2d, layer=0,
                 use_dropout=False, conv2_layer_count=11):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()

        self.attention = None
        if conv2_layer_count==11:
            self.attention = SelfAttention(512) # 初始化 self-attention 層
            self.res_block3 = ResidualBlock(512, 512) # 第 3 層的輸出通道數為 512
            self.res_block5 = ResidualBlock(256, 256) # 第 5 層的輸出通道數為 256

        outermost=False
        innermost=False
        if layer==1:
            innermost=True
        if layer==8:
            outermost=True
        self.outermost = outermost
        self.innermost = innermost
        self.layer = layer
        self.conv2_layer_count = conv2_layer_count

        use_bias = True
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        #self.model = nn.Sequential(*model)
        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x, style=None):
        if self.outermost:
            down_out = self.down(x)
            if style is None:
                return self.submodule(down_out)
            sub, down_out = self.submodule(down_out, style)
            out = self.up(sub)
            return out, down_out
        else:
            out = None
            down_out = self.down(x)
            if self.innermost:
                if style is None:
                    return down_out
                out = self.up(down_out)
                return torch.cat([x, out], 1), down_out.view(x.shape[0], -1)
            else:
                if style is None:
                    return self.submodule(down_out)
                sub, down_out = self.submodule(down_out, style)
                out = self.up(sub)
                if self.conv2_layer_count == 11:
                    if self.layer == 4:
                        out = self.attention(out)
                    if self.layer == 3:
                        out = self.res_block3(out) # 在第 3 層之後加入殘差塊
                    if self.layer == 5:
                        out = self.res_block5(out) # 在第 5 層之後加入殘差塊
                return torch.cat([x, out], 1), down_out



# 自注意力機制
class SelfAttention(nn.Module):
    """ Self-Attention 層 """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        # 查詢、鍵、值的卷積層
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 輸出卷積層
        self.gamma = nn.Parameter(torch.zeros(1))  # 可學習的縮放參數

    def forward(self, x):
        batch_size, C, height, width = x.size()
        # 計算查詢、鍵、值
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(batch_size, -1, height * width)  # (B, C//8, H*W)
        value = self.value(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # 計算注意力分數
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = nn.functional.softmax(attention, dim=-1)  # 沿最後一維做 softmax

        # 加權求和
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, height, width)  # 恢復形狀

        # 加入殘差連接
        out = self.gamma * out + x
        return out


# 殘差塊
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out