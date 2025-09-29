from torch import nn


class TinyVGG(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        input_shape: int,  # number of channels
        output_shape: int,  # number of classes
        batch_norm: bool = False,
        dropout: float = 0.0,  # if 0.0, no dropout
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout

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
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.conv_block1 = conv_block(input_shape, hidden_units)
        self.conv_block2 = conv_block(hidden_units, hidden_units)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.LazyLinear(out_features=output_shape),
        )

    def forward(self, X):
        X = self.conv_block1(X)
        X = self.conv_block2(X)
        X = self.classifier(X)
        return X
