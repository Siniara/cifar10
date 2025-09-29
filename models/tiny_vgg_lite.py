"""
Original TinyVGG implementation adapted from https://poloclub.github.io/cnn-explainer/.
This version has no regularization and is provided for reference.

The version used in the project (models/tiny_vgg.py) supports dropout and batch normalization,
while still allowing this plain architecture.
"""

from torch import nn


class TinyVGG(nn.Module):
    def __init__(self, hidden_units: int, input_shape: int, output_shape: int):
        """TinyVGG model adapted from https://poloclub.github.io/cnn-explainer/.
        Args:
            hidden_units: Number of hidden units between layers.
            input_shape: Number of channels in the input data. For example, 3 for RGB images, 1 for grayscale.
            output_shape: Number of output classes.
        """
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=output_shape),
        )

    def forward(self, X):
        X = self.conv_block1(X)
        X = self.conv_block2(X)
        X = self.classifier(X)
        return X
