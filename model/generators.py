import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_q = self.query_conv(x).view(b, -1, h*w).permute(0, 2, 1)   # (B, HW, d_k)
        proj_k = self.key_conv(x).view(b, -1, h*w)                     # (B, d_k, HW)
        d_k = proj_q.size(-1)
        energy = torch.bmm(proj_q, proj_k) / math.sqrt(max(1, d_k))
        attention = F.softmax(energy, dim=-1)                          # (B, HW, HW)

        proj_v = self.value_conv(x).view(b, c, h*w)                    # (B, C, HW)
        out = torch.bmm(proj_v, attention.permute(0, 2, 1))            # (B, C, HW)
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 capture_second=False,    # <- set True for second-innermost to capture its enc map
                 embedding_dim=128, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, use_pixel_shuffle=False, use_res_skip=False,
                 self_attention=False):
        """
        capture_second: if True, this block will return its `enc` as the 'second_deepest' feature.
        The innermost block will return the deepest encoding.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.capture_second = capture_second
        self.use_res_skip = use_res_skip
        self.self_attention = self_attention

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        # down / up ops
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # attention (applied on the encoded map)
        if self_attention:
            # apply attention to encode (inner_nc channels)
            self.att = SelfAttention(inner_nc)
        else:
            self.att = None

        # compute up_in channels
        if innermost:
            up_in_nc = inner_nc + embedding_dim
        else:
            up_in_nc = inner_nc if use_res_skip else inner_nc * 2

        # choose pixelshuffle or transpose conv path
        if outermost:
            if use_pixel_shuffle:
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1)
                up = [uprelu, upconv, nn.PixelShuffle(2), nn.Tanh()]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc, kernel_size=4, stride=2, padding=1)
                up = [uprelu, upconv, nn.Tanh()]
            down = [downconv]
        elif innermost:
            if use_pixel_shuffle:
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
                up = [uprelu, upconv, nn.PixelShuffle(2), upnorm]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
                up = [uprelu, upconv, upnorm]
            down = [downrelu, downconv]
        else:
            if use_pixel_shuffle:
                upconv = nn.Conv2d(up_in_nc, outer_nc * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
                up = [uprelu, upconv, nn.PixelShuffle(2), upnorm]
            else:
                upconv = nn.ConvTranspose2d(up_in_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
                up = [uprelu, upconv, upnorm]
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x, style=None):
        """
        Returns tuple depending on role:
        - innermost:
            if style is provided -> returns (dec_out, deepest_feat, None_or_second)
            else -> returns encode (for encode-only mode)
        - non-outermost:
            returns (combined_feature, deepest_feat, second_feat)
        - outermost:
            returns (image_out, deepest_feat, second_feat)
        Notes: deepest_feat and second_feat are feature maps with spatial dims.
        """
        if self.innermost:
            # encode
            encode = self.down(x)  # (B, inner_nc, h, w)
            if self.att is not None:
                encode = self.att(encode)
            if style is None:
                # encode-only mode
                return encode
            # concat style embedding
            style_b = style.view(style.shape[0], style.shape[1], 1, 1)
            style_b = style_b.expand(-1, -1, encode.size(2), encode.size(3))
            enc = torch.cat([style_b, encode], 1)  # (B, inner_nc + embed_dim, h, w)
            dec = self.up(enc)
            if self.use_res_skip:
                out = x + dec
            else:
                out = torch.cat([x, dec], 1)
            # innermost returns deepest encoding as 'deepest_feat'
            return out, encode, None

        elif self.outermost:
            enc = self.down(x)
            if self.att is not None:
                enc = self.att(enc)
            # propagate into deeper submodule (should return sub_out, deepest_feat, second_feat)
            sub_out = self.submodule(enc, style) if style is None else self.submodule(enc, style)
            # sub_out can be either tensor (encode-only) or tuple
            if isinstance(sub_out, tuple):
                sub, deepest_feat, second_feat = sub_out
            else:
                # if style==None path and deeper returned only encode
                sub = sub_out
                deepest_feat = None
                second_feat = None
            dec = self.up(sub)
            if self.att is not None:
                dec = self.att(dec)
            # outermost returns final image and the two feature maps
            return dec, deepest_feat, second_feat

        else:
            # normal intermediate block
            enc = self.down(x)  # encoded small map
            if self.att is not None:
                enc = self.att(enc)
            # call deeper
            sub_out = self.submodule(enc, style) if style is None else self.submodule(enc, style)
            if isinstance(sub_out, tuple):
                sub, deepest_feat, second_feat = sub_out
            else:
                sub = sub_out
                deepest_feat = None
                second_feat = None

            # if this block is designated as second-deepest, capture enc as second_feat
            if self.capture_second:
                # enc is the second-deepest encoded map (B, inner_nc, h, w)
                second_feat = enc

            dec = self.up(sub)
            if self.att is not None:
                dec = self.att(dec)

            if self.use_res_skip:
                combined = x + dec
            else:
                combined = torch.cat([x, dec], 1)

            return combined, deepest_feat, second_feat

# --------------------------------
# UNet Generator (builder) - Option B
# --------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64,
                 embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 self_attention=True, use_pixel_shuffle=True, use_res_skip=True):
        super(UNetGenerator, self).__init__()
        self.embedder = nn.Embedding(embedding_num, embedding_dim)
        self.use_res_skip = use_res_skip

        # Build innermost with attention (bottleneck)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             input_nc=None, submodule=None,
                                             innermost=True, embedding_dim=embedding_dim,
                                             norm_layer=norm_layer, use_dropout=use_dropout,
                                             use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip,
                                             self_attention=True, capture_second=False)

        # add several ngf*8 blocks (no attention)
        for _ in range(num_downs - 6):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                                 input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip,
                                                 self_attention=False, capture_second=False)

        # second-innermost: capture this enc as second_deepest and add attention here
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,
                                             input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout,
                                             use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip,
                                             self_attention=True, capture_second=True)

        # continue building up
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8,
                                             input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle,
                                             use_res_skip=use_res_skip, self_attention=False, capture_second=False)

        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4,
                                             input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle,
                                             use_res_skip=use_res_skip, self_attention=False, capture_second=False)

        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,
                                             input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_pixel_shuffle=use_pixel_shuffle,
                                             use_res_skip=use_res_skip, self_attention=False, capture_second=False)

        # outermost
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer,
                                             use_pixel_shuffle=use_pixel_shuffle, use_res_skip=use_res_skip,
                                             self_attention=False, capture_second=False)

    def forward(self, x, style_or_label=None, return_feat=False):
        """
        If return_feat=True: returns (out_image, feat) where feat = concat(deepest, second_deepest)
        else: returns out_image only.
        style_or_label: either LongTensor (labels) or precomputed embedding (B, embedding_dim)
        """
        if style_or_label is not None and hasattr(style_or_label, "type") and 'LongTensor' in style_or_label.type():
            style = self.embedder(style_or_label)
        else:
            style = style_or_label

        # model returns (image, deepest_feat, second_feat)
        out_tuple = self.model(x, style)
        if isinstance(out_tuple, tuple):
            out_img, deepest_feat, second_feat = out_tuple
        else:
            # when style is None and model returned encode-only (unlikely with outermost)
            return out_tuple

        if return_feat:
            if deepest_feat is None or second_feat is None:
                # safety fallback: if one missing, return whichever is available
                if deepest_feat is None:
                    feat = second_feat
                elif second_feat is None:
                    feat = deepest_feat
                else:
                    feat = None
            else:
                # concat along channel dim

                Hd, Wd = deepest_feat.shape[2:]
                Hs, Ws = second_feat.shape[2:]

                if (Hd != Hs) or (Wd != Ws):
                    second_feat = F.interpolate(second_feat, size=(Hd, Wd), mode="nearest")

                feat = torch.cat([deepest_feat, second_feat], dim=1)
            return out_img, feat
        else:
            return out_img

# -----------------------
# quick test (sanity)
# -----------------------
if __name__ == "__main__":
    device = torch.device("cpu")
    netG = UNetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64,
                         embedding_num=40, embedding_dim=128,
                         norm_layer=nn.BatchNorm2d, use_dropout=False,
                         self_attention=True, use_pixel_shuffle=True, use_res_skip=True).to(device)
    print(netG)
    x = torch.randn(2, 1, 256, 256).to(device)
    labels = torch.LongTensor([0, 1]).to(device)
    y, feat = netG(x, labels, return_feat=True)
    print("out shape:", y.shape)     # expect (2, 1, 256, 256)
    print("feat shape:", feat.shape) # expect (2, 1024, Hsmall, Wsmall) e.g. (2,1024,4,4) depending on num_downs
