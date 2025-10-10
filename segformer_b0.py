#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class MixFFN(nn.Module):
        """Mix-FFN module inspired by SegFormer"""
        def __init__(self, in_ch, hidden_ch):
                super().__init__()
                self.fc1 = nn.Conv2d(in_ch, hidden_ch, 1)
                self.dwconv = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=hidden_ch)
                self.fc2 = nn.Conv2d(hidden_ch, in_ch, 1)

        def forward(self, x):
                x = self.fc1(x)
                x = self.dwconv(x)
                x = F.gelu(x)
                x = self.fc2(x)
                return x


class SegFormerEncoderBlock(nn.Module):
        """Lightweight encoder block inspired by SegFormer's MiT"""
        def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.proj = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU()
                )
                self.attn = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU()
                )
                self.ffn = MixFFN(out_ch, out_ch * 4)
                self.norm1 = nn.BatchNorm2d(out_ch)
                self.norm2 = nn.BatchNorm2d(out_ch)

        def forward(self, x):
                x = self.proj(x)
                # Attention-like operation
                x = x + self.attn(self.norm1(x))
                # FFN
                x = x + self.ffn(self.norm2(x))
                return x


class SegFormerB0(nn.Module):
        def __init__(self, num_classes, in_ch=1, proj_dim=128, **kwargs):
                super().__init__()

                # Lightweight hierarchical encoder (inspired by MiT-B0)
                # Creates 4 stages with progressively increasing channels
                self.stage1 = nn.Sequential(
                        SegFormerEncoderBlock(in_ch, 32, stride=2),
                        SegFormerEncoderBlock(32, 32, stride=1)
                )
                self.stage2 = nn.Sequential(
                        SegFormerEncoderBlock(32, 64, stride=2),
                        SegFormerEncoderBlock(64, 64, stride=1)
                )
                self.stage3 = nn.Sequential(
                        SegFormerEncoderBlock(64, 160, stride=2),
                        SegFormerEncoderBlock(160, 160, stride=1)
                )
                self.stage4 = nn.Sequential(
                        SegFormerEncoderBlock(160, 256, stride=2),
                        SegFormerEncoderBlock(256, 256, stride=1)
                )

                # MLP decoder head
                chans = [32, 64, 160, 256]
                self.proj = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in chans])
                self.fuse = nn.Conv2d(proj_dim * 4, 256, 1)
                self.head = nn.Conv2d(256, num_classes, 1)

                print(f"> Initialized {self.__class__.__name__} (in_ch={in_ch}->num_classes={num_classes}) with {kwargs}")

        def forward(self, x):
                B, C, H, W = x.shape

                # Multi-scale feature extraction
                x1 = self.stage1(x)     # stride 2
                x2 = self.stage2(x1)    # stride 4
                x3 = self.stage3(x2)    # stride 8
                x4 = self.stage4(x3)    # stride 16

                feats = [x1, x2, x3, x4]

                # All-MLP decoder
                ups = []
                for f, p in zip(feats, self.proj):
                        u = p(f)
                        u = F.interpolate(u, size=(H, W), mode="bilinear", align_corners=False)
                        ups.append(u)

                fused = torch.cat(ups, dim=1)              # B, 4*proj_dim, H, W
                fused = F.relu(self.fuse(fused), inplace=True)
                logits = self.head(fused)                  # B, num_classes, H, W
                return logits

        def init_weights(self, *args, **kwargs):
                # Apply weight initialization to all modules
                self.apply(random_weights_init)
