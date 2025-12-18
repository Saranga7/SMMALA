import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, name: str, num_classes: int, dropout: float = 0.5):
        super().__init__()

        assert any(k in name for k in ["small", "medium", "big"]), \
            "name must contain 'small', 'medium', or 'big'"

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------------------------------
        # Architecture definition
        # -------------------------------------------------
        if "small" in name:
            channels = [3, 16, 32]
            hidden_dim = 64
            mlp_dims = [128, 64]              # 2 hidden layers
        elif "medium" in name:
            channels = [3, 16, 32, 64]
            hidden_dim = 128
            mlp_dims = [256, 128, 128]        # 3 hidden layers
        else:  # big
            channels = [3, 16, 32, 64, 128]
            hidden_dim = 256
            mlp_dims = [512, 256, 128, 64]   # 4 hidden layers

        # -------------------------------------------------
        # Convolutional blocks
        # -------------------------------------------------
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
            for i in range(len(channels) - 1)
        ])

        # -------------------------------------------------
        # Feature projection
        # -------------------------------------------------
        self.fc1 = nn.Linear(channels[-1], hidden_dim)

        # -------------------------------------------------
        # MLP classification head
        # -------------------------------------------------
        self.out_features = num_classes if num_classes > 2 else 1

        mlp_layers = []
        in_dim = hidden_dim
        for dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, dim),
                nn.LayerNorm(dim),
                nn.PReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim

        mlp_layers.append(nn.Linear(in_dim, self.out_features))
        self.fc = nn.Sequential(*mlp_layers)

        # -------------------------------------------------
        # Feature interface
        # -------------------------------------------------
        self.feature_dim = hidden_dim
        self.use_features = False

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))

        x = self.gap(x)              # [B, C, 1, 1]
        x = x.flatten(1)             # [B, C]

        features = F.relu(self.fc1(x))  # [B, feature_dim]

        if self.use_features:
            return features

        # Optional 
        features = F.normalize(features, dim = 1)

        return self.fc(features)

    # -------------------------------------------------
    # Raw conv features (optional)
    # -------------------------------------------------
    def features_extraction(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        return x

    @property
    def num_features(self):
        """for compatibility with SlideAggregator"""
        return self.feature_dim

