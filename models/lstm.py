import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM Model for stock price prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_size: Number of output predictions
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

