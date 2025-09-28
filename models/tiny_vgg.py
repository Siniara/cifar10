from torch import nn


class TinyVGG(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        input_shape: int,
        output_shape: int,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.batch_norm = batch_norm

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.conv_block1 = conv_block(input_shape, hidden_units)
        self.conv_block2 = conv_block(hidden_units, hidden_units)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(out_features=output_shape)
        )

    def forward(self, X):
        X = self.conv_block1(X)
        X = self.conv_block2(X)
        X = self.classifier(X)
        return X
