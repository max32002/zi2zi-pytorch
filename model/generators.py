import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, self_attention=False):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        # Configurations requested by user
        use_pixel_shuffle = True
        use_res_skip = True

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, embedding_dim=embedding_dim, 
                                             use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip)  # add the innermost layer
        
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        # Add SelfAttention to ngf*4 and ngf*2 blocks
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle, 
                                             use_res_skip=use_res_skip, self_attention=self_attention)
        
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle, 
                                             use_res_skip=use_res_skip, self_attention=self_attention)
        
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle, 
                                             use_res_skip=use_res_skip)
        
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle, 
                                             use_res_skip=use_res_skip)  # add the outermost layer
        
        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x, style_or_label=None):
        """Standard forward"""
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            return self.model(x, self.embedder(style_or_label))
        else:
            return self.model(x, style_or_label)


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, embedding_dim=128, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, use_pixel_shuffle=False, use_res_skip=False, self_attention=False):
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
        self.outermost = outermost
        self.innermost = innermost
        self.use_res_skip = use_res_skip
        self.self_attention = self_attention
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if self_attention:
            self.att = SelfAttention(outer_nc)

        # Determine input channels for upsampling
        if innermost:
            up_in_nc = inner_nc + embedding_dim
        else:
            # For outermost and intermediate, input comes from submodule.
            # If submodule used ResSkip, it returns inner_nc channels.
            # If submodule used Concat, it returns inner_nc * 2 channels.
            up_in_nc = inner_nc if use_res_skip else inner_nc * 2

        if outermost:
            if use_pixel_shuffle:
                # Output is outer_nc * 4 before PS to result in outer_nc
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1)
                up = [uprelu, upconv, nn.PixelShuffle(2), nn.Tanh()]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                up = [uprelu, upconv, nn.Tanh()]
            down = [downconv]

        elif innermost:
            if use_pixel_shuffle:
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
                up = [uprelu, upconv, nn.PixelShuffle(2), upnorm]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                up = [uprelu, upconv, upnorm]
            down = [downrelu, downconv]

        else:
            if use_pixel_shuffle:
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
                up = [uprelu, upconv, nn.PixelShuffle(2), upnorm]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                up = [uprelu, upconv, upnorm]
            
            down = [downrelu, downconv, downnorm]

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
            
            # Robust style expansion for 1x1 or 3x3 bottlenecks
            style_b = style.view(style.shape[0], style.shape[1], 1, 1)
            style_b = style_b.expand(-1, -1, encode.size(2), encode.size(3))
            enc = torch.cat([style_b, encode], 1)
            
            dec = self.up(enc)
            
            if self.use_res_skip:
                return x + dec, encode.view(x.shape[0], -1)
            else:
                return torch.cat([x, dec], 1), encode.view(x.shape[0], -1)
        
        elif self.outermost:
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            if self.self_attention:
                dec = self.att(dec)
            return dec, encode
        
        else:  # add skip connections
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            
            if self.self_attention:
                dec = self.att(dec)
                
            if self.use_res_skip:
                # Add skip connection
                return x + dec, encode
            else:
                # Concat skip connection
                return torch.cat([x, dec], 1), encode
