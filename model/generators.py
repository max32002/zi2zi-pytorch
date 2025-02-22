import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, self_attention=False, self_attention_layer=4, residual_block=False, residual_block_layer=[]):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            self_attention  -- self attention status
            self_attention_layer -- append to layer
            residual_block  -- residual block status
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure
        
        print("UNetGenerator self_attention", self_attention)
        print("UNetGenerator residual_block", residual_block)

        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, layer=1, embedding_dim=embedding_dim, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        for index in range(num_downs - 5):  # add intermediate layers with ngf * 8 filtersv
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=index+2, use_dropout=use_dropout, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=5, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=6, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=7, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, layer=8, self_attention=self_attention, self_attention_layer=self_attention_layer, residual_block=residual_block, residual_block_layer=residual_block_layer)
        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x, style_or_label=None):
        """Standard forward"""
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            style=self.embedder(style_or_label)
            return self.model(x, style)
        else:
            return self.model(x, style_or_label)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, embedding_dim=128, norm_layer=nn.BatchNorm2d, layer=0,
                 use_dropout=False, self_attention=False, self_attention_layer=4, residual_block=False, residual_block_layer=[]):
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
        self.attn = None
        if self_attention:
            self.attn = SelfAttention(512) # 初始化 self-attention 層
        self.res_block3 = None
        self.res_block5 = None
        if residual_block:
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
        self.self_attention = self_attention
        self.self_attention_layer = self_attention_layer
        self.residual_block = residual_block
        self.residual_block_layer = residual_block_layer
        self.embedding_dim = embedding_dim
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        print("layer", layer)
        print("self_attention", self_attention)
        print("residual_block", residual_block)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x, style=None):
        if self.innermost:
            encode = self.down(x)
            if style is None:
                return encode
            up_input = None
            if self.embedding_dim > 0:
                new_style = style.view(style.shape[0], self.embedding_dim, 1, 1)
                if encode.shape[2] != new_style.shape[2]:
                    new_style = nn.functional.interpolate(new_style, size=[encode.size(2), encode.size(3)], mode='bilinear', align_corners=False)
                up_input = torch.cat([new_style, encode], 1)
            else:
                up_input = encode
            dec = self.up(up_input)
            dec_resized = dec
            if x.shape[2] != dec.shape[2]:
                dec_resized = nn.functional.interpolate(dec, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
            ret1=torch.cat([x, dec_resized], 1)
            ret2=encode.view(x.shape[0], -1)
            return ret1, ret2

        elif self.outermost:
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            up_input, encode = self.submodule(enc, style)
            dec = self.up(up_input)
            return dec, encode
        else:  # add skip connections
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            up_input, encode = self.submodule(enc, style)
            dec = self.up(up_input)
            if self.self_attention:
                if self.layer == self.self_attention_layer:
                    dec = self.attn(dec) # 加入 self-attention 層
            
            if self.residual_block:
                if self.layer == 3:
                    dec = self.res_block3(dec) # 在第 3 層之後加入殘差塊
                if self.layer == 5:
                    dec = self.res_block5(dec) # 在第 5 層之後加入殘差塊
            
            dec_resized = dec
            if x.shape[2] != dec.shape[2]:
                dec_resized = nn.functional.interpolate(dec, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
            ret1=torch.cat([x, dec_resized], 1)

            return ret1, encode

# 自注意力機制
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = nn.functional.softmax(attention, dim=-1)
        attention = torch.bmm(value, attention.permute(0, 2, 1))
        attention = attention.view(batch_size, channels, height, width)

        return x + self.gamma * attention

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