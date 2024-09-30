import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False


class LinearDeepNN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, seed=42):
        super(LinearDeepNN, self).__init__()
        set_seed(seed)  # Set the seed for reproducibility

        # Define the layers
        self.fc1 = nn.Linear(np.prod(input_dims), 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, n_actions)  # Hidden layer to output layer

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Flatten the grid into a single vector
        flat_state = state.view(state.size()[0], -1)
        x = F.relu(self.fc1(flat_state))  # Pass through the first fully connected layer
        x = F.relu(self.fc2(x))  # Pass through the second fully connected layer
        actions = self.fc3(x)  # Output the action values

        return actions

    def print_structure(self):
        print("Model Structure: CNN" if isinstance(self, ConvolutionalDeepNN) else "Model Structure: Linear NN")
        print("=" * 60)

        # Iterate over all named children (layers) of the network
        for name, layer in self.named_children():
            # Print the layer name and type
            print(f"Layer: {name} | Type: {layer.__class__.__name__}")

            # If it's a Linear layer, print input/output features
            if isinstance(layer, nn.Linear):
                print(f"  Input Features: {layer.in_features} | Output Features: {layer.out_features}")

            # If it's a Convolutional layer, print kernel size and stride details
            elif isinstance(layer, nn.Conv2d):
                print(f"  In Channels: {layer.in_channels} | Out Channels: {layer.out_channels}")
                print(f"  Kernel Size: {layer.kernel_size} | Stride: {layer.stride}")

            # Print if the layer has bias terms or not
            if hasattr(layer, 'bias') and layer.bias is not None:
                print(f"  Bias: Yes | Bias Shape: {layer.bias.shape}")
            else:
                print("  Bias: No")

            # Print a separator for readability
            print("-" * 60)

        # Print the loss function used
        print("Loss Function: ", self.loss.__class__.__name__)

        # Print the optimizer details
        print("Optimizer: ", self.optimizer.__class__.__name__)
        print("Learning Rate: ", self.optimizer.param_groups[0]['lr'])

        # Print device information
        print(f"Device: {self.device}")

        print("=" * 60)


class ConvolutionalDeepNN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, seed=42):
        super(ConvolutionalDeepNN, self).__init__()  # Super constructor call
        set_seed(seed)  # Set the seed for reproducibility

        # Convolutional Layers - input_dims[0] refers to channels this example has 4 images stacked in grayscale
        #                        so this will equal (4,1).
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Function to help us find fully connected layer input dims
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)  # RMSProp optimizer

        self.loss = nn.MSELoss()  # MSE Loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # Check device available
        self.to(self.device)  # and send if needed.

    def calculate_conv_output_dims(self, input_dims):
        """This function simply ensures that the input to the fully connected layer is the right shape."""
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)  # Pass through conv layer 1.
        dims = self.conv2(dims)  # Pass through conv layer 2.
        dims = self.conv3(dims)  # Pass through conv layer 3.
        # We need to multiply out the dimensions returned here to get out input to the first fully connected layer.
        dims_prod = int(np.prod(dims.size()))
        return dims_prod

    def forward(self, state):
        """Input: state of environment as BatchSize x input_dims.
                        Output: action values for given state or batch of states.
                        Note agnostic to dimensions (batch-size etc.) so long as we are consistent."""
        conv1 = F.relu(self.conv1(state))  # Pass state through first conv layer and activate with Relu.
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        # Reshape to batch size x number of input features of our fc1 using view function
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions  # Final output is the action values for the given state or batch of states.

    def print_structure(self):
        print("Model Structure: CNN" if isinstance(self, ConvolutionalDeepNN) else "Model Structure: Linear NN")
        print("=" * 60)

        # Iterate over all named children (layers) of the network
        for name, layer in self.named_children():
            # Print the layer name and type
            print(f"Layer: {name} | Type: {layer.__class__.__name__}")

            # If it's a Linear layer, print input/output features
            if isinstance(layer, nn.Linear):
                print(f"  Input Features: {layer.in_features} | Output Features: {layer.out_features}")

            # If it's a Convolutional layer, print kernel size and stride details
            elif isinstance(layer, nn.Conv2d):
                print(f"  In Channels: {layer.in_channels} | Out Channels: {layer.out_channels}")
                print(f"  Kernel Size: {layer.kernel_size} | Stride: {layer.stride}")

            # Print if the layer has bias terms or not
            if hasattr(layer, 'bias') and layer.bias is not None:
                print(f"  Bias: Yes | Bias Shape: {layer.bias.shape}")
            else:
                print("  Bias: No")

            # Print a separator for readability
            print("-" * 60)

        # Print the loss function used
        print("Loss Function: ", self.loss.__class__.__name__)

        # Print the optimizer details
        print("Optimizer: ", self.optimizer.__class__.__name__)
        print("Learning Rate: ", self.optimizer.param_groups[0]['lr'])

        # Print device information
        print(f"Device: {self.device}")

        print("=" * 60)
