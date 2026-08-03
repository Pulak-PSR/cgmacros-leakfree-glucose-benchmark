# Model Implementations

## LSTM Model Implementation
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

## Transformer Model Implementation
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(input_dim, num_heads, num_layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        out = self.transformer(x)
        out = self.fc(out)
        return out

## Ensemble Model Implementation
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return sum(outputs) / len(outputs)  # Average the outputs

# Example usage
# lstm = LSTMModel(input_size=10, hidden_size=20, num_layers=2)
# transformer = TransformerModel(input_dim=10, model_dim=20, num_heads=2, num_layers=2)
# ensemble = EnsembleModel(models=[lstm, transformer])
