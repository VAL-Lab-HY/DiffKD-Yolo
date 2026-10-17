from torch import nn


class LightConditionEncoder(nn.Module):
    """Predict a full-resolution illumination map ``L`` in the range [0, 1]."""

    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, images):
        return self.encoder(images)
