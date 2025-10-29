"""Seq-to-seq DTA-SNN variant with a fully spiking encoder."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import (
    BasicBlock_MS,
    tdLayer,
    tdBatchNorm,
    LIFSpike,
    add_dimention,
    conv1x1,
)
from models.DTA import DTA


class MSResNetEncoder(nn.Module):
    """Spiking ResNet encoder that mirrors the classifier backbone."""

    def __init__(
        self,
        block: nn.Module = BasicBlock_MS,
        layers: tuple = (3, 3, 2),
        time_step: int = 6,
        DTA_ON: bool = True,
        dvs: bool = True,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.dvs = dvs
        self.T = time_step
        self._norm_layer = tdBatchNorm
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Ensure the very first convolution also receives spiking inputs.
        self.input_spike = LIFSpike()
        self.input_conv = tdLayer(
            nn.Conv2d(
                in_channels,
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self._norm_layer(self.inplanes),
        )

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)

        self.LIF = LIFSpike()
        self.encoding = DTA(T=self.T, out_channels=64) if DTA_ON else None

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers_list = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dvs:
            x = add_dimention(x, self.T)
        x = self.input_spike(x)
        x = self.input_conv(x)
        if self.encoding is not None:
            analog = x
            spikes = self.LIF(x)
            x = self.encoding(analog, spikes)
        else:
            x = self.LIF(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.LIF(x)
        return x


class DTASeq2Seq(nn.Module):
    """Future-frame predictor built on top of the spiking encoder."""

    def __init__(
        self,
        pre_seq: int,
        aft_seq: int,
        in_channels: int = 2,
        out_channels: int = 2,
        encoder_time: Optional[int] = None,
        encoder_layers: tuple = (3, 3, 2),
        bottleneck_channels: int = 256,
        DTA_ON: bool = True,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        encoder_time = encoder_time or pre_seq
        self.pre_seq = pre_seq
        self.aft_seq = aft_seq
        self.height: Optional[int] = None
        self.width: Optional[int] = None

        self.encoder = MSResNetEncoder(
            block=BasicBlock_MS,
            layers=encoder_layers,
            time_step=encoder_time,
            DTA_ON=DTA_ON,
            dvs=True,
            in_channels=in_channels,
        )
        encoder_channels = 512 * BasicBlock_MS.expansion
        self.projection = nn.Conv3d(encoder_channels, bottleneck_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1),
        )
        self.output_activation = nn.Sigmoid() if activation == "sigmoid" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input of shape [batch, time, channels, height, width].")
        if x.shape[1] != self.pre_seq:
            raise ValueError(f"Got {x.shape[1]} input steps, expected {self.pre_seq}.")
        self.height, self.width = x.shape[-2], x.shape[-1]
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1, 3, 4).contiguous()
        encoded = self.projection(encoded)
        encoded = F.interpolate(
            encoded,
            size=(self.aft_seq, self.height, self.width),
            mode="trilinear",
            align_corners=False,
        )
        decoded = self.decoder(encoded)
        decoded = self.output_activation(decoded)
        return decoded.permute(0, 2, 1, 3, 4).contiguous()


def build_dta_snn_seq2seq(**kwargs) -> DTASeq2Seq:
    return DTASeq2Seq(**kwargs)
