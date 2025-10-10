#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d((2,2)) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class CRNNEvent(nn.Module):
    def __init__(self, n_mels=64, rnn_hidden=128, rnn_layers=1, num_classes=1):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(
            input_size=128 * (n_mels // 8),  # after 3x pool(2,2)
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        h = self.backbone(x)          # (B, C, F', T')
        B, C, Fp, Tp = h.shape
        h = self.dropout(h)
        # collapse freq to feature, time as sequence
        h = h.permute(0, 3, 1, 2).contiguous()  # (B, T', C, F')
        h = h.view(B, Tp, C * Fp)               # (B, T', Feat)
        h, _ = self.rnn(h)                      # (B, T', 2H)
        h = h.mean(dim=1)                       # time-mean pooling
        logits = self.head(h)                   # (B, 1)
        return logits.squeeze(1)
