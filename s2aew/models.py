import torch
import torch.nn as nn
from torch.nn import functional as F


class HamidaEtAl(nn.Module):
    """3-D CNN for HSI classification after Hamida et al., TGRS 2018."""

    def __init__(self, input_channels: int, n_classes: int, patch_size: int = 11, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=1, dilation=dilation, padding=0)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), dilation=dilation, stride=1, padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=1, padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.fc(x.view(-1, self.features_size))


class ChenEtAl(nn.Module):
    """3-D CNN for HSI classification after Chen et al., TGRS 2017."""

    def __init__(self, input_channels: int, n_classes: int, patch_size: int = 11, n_planes: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(3, 1, 1))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (5, 3, 3), padding=(2, 1, 1))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (3, 3, 3), padding=(1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x.view(-1, self.features_size))
        return self.fc(x)


class HeEtAl(nn.Module):
    """Multi-scale 3-D CNN for HSI classification after He et al., ICIP 2017."""

    def __init__(self, input_channels: int, n_classes: int, patch_size: int = 11):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.dropout = nn.Dropout(p=0.6)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)
            x = self.conv3_1(x) + self.conv3_2(x) + self.conv3_3(x) + self.conv3_4(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x))
        x = F.relu(self.conv3_1(x) + self.conv3_2(x) + self.conv3_3(x) + self.conv3_4(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x.view(-1, self.features_size))
        return self.fc(x)
