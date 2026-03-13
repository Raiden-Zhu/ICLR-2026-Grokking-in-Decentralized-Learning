import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,  # 28x28 for MNIST
        hidden_sizes: list = [1024, 512],
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        use_bn: bool = True,
        activation: nn.Module = nn.ReLU
    ) -> None:
        super(MLP, self).__init__()
        
        # Input layer -> First hidden layer
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(activation(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(activation(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

def create_mlp(input_size: int = 12288, num_classes: int = 200, **kwargs) -> MLP:
    """
    Factory function to create MLP instance with default or custom parameters
    
    Args:
        input_size: Input dimension size
        num_classes: Number of output classes
        **kwargs: Additional arguments to pass to MLP constructor
    
    Returns:
        MLP model instance
    """
    return MLP(input_size=input_size, num_classes=num_classes, **kwargs)