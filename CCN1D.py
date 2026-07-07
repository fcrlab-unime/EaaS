import torch

class CCN1D(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int, kernel_size=3, dropout=0.1, output_size: int = 1):
        """
        1D Convolutional Neural Network (CNN) module for sequence data.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of channels in the hidden layers.
            num_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernel. Defaults to 3.
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(CCN1D, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()

        # First convolutional layer
        self.conv_layers.append(torch.nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Dynamically add convolutional layers
        for i in range(1, num_layers):
            layer_channels = hidden_channels // (2 ** i)
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels // (2 ** (i - 1)), layer_channels, kernel_size, padding=kernel_size//2))
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(layer_channels))

        # Output layer
        self.output_layer =  torch.nn.Conv1d(hidden_channels // (2 ** (num_layers - 1)), output_size, kernel_size, padding=kernel_size//2)

        # Activation and Dropout
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, sequence_length).
        """
        for conv_layer, batch_norm_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = self.relu(batch_norm_layer(conv_layer(x)))
            x = self.dropout(x)

        x = self.output_layer(x)
        x = x.squeeze()
        return x
