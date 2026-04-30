import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_size
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            from torchvision.models import resnet18

            backbone = resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def create_model(model_name: str, num_classes: int, pretrained: bool = True, mlp_input_size: int | None = None, mlp_hidden_dims: list[int] | None = None, mlp_dropout: float = 0.2) -> nn.Module:
    name = (model_name or "").lower()
    if name in {"resnet18", "resnet-18"}:
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    if name in {"mlp"}:
        if mlp_input_size is None or mlp_hidden_dims is None:
            raise ValueError("MLP requires mlp_input_size and mlp_hidden_dims")
        return MLP(mlp_input_size, mlp_hidden_dims, num_classes, dropout_rate=mlp_dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
