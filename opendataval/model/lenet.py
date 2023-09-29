import torch
import torch.nn as nn
import torch.nn.functional as F

from opendataval.model.api import TorchClassMixin, TorchPredictMixin


class LeNet(TorchClassMixin, TorchPredictMixin):
    """LeNet-5 convolutional neural net classifier.

    Consists of 2 5x5 convolution kernels and a MLP classifier. LeNet-5 was one of the
    earliest conceived CNNs and was typically applied to digit analysis. LeNet-5 can but
    doesn't generalize particularly well to higher dimension (such as color) images.

    References
    ----------
    .. [1] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner,
        Gradient-based learning applied to document recognition,
        Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998,
        doi: https://doi.org/10.1109/5.726791.

    Parameters
    ----------
    num_classes : int
        Number of prediction classes
    gray_scale : bool, optional
        Whether the input image is gray scaled. LeNet has been noted to not perform
        as well with color, so disable gray_scale at your own risk, by default True
    """

    def __init__(self, num_classes: int, gray_scale: bool = True):

        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 kernel
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1 if gray_scale else 3, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.LazyLinear(120)  # Automatically sets input to output of max_pool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor):
        """Forward pass of LeNet-5."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # Max pooling over a (2, 2) window

        # Second CNN
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # MLP
        x = torch.flatten(x, start_dim=1)  # flatten all dimensions
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, -1)

        return x
