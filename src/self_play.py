"""
Policy Network for Othello AlphaZero

This file contains the neural network that serves as the "brain" of our AI.
The network learns to:
1. Evaluate which moves are good (Policy)
2. Predict who will win (Value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OthelloPolicyNetwork(nn.Module):
    """
    Neural Network for Othello using Convolutional layers.
    
    Architecture similar to AlphaGo/AlphaZero but simpler.
    
    Input: 8x8 board (1 = our pieces, -1 = opponent, 0 = empty)
    Output: 
        - Policy: 64 probabilities (one per square)
        - Value: Single number (-1 to +1, negative = losing, positive = winning)
    """
    
    def __init__(self, num_filters=128, num_residual_blocks=4):
        """
        Initialize the network.
        
        Args:
            num_filters: Number of convolutional filters (wider = more capacity)
            num_residual_blocks: Number of residual blocks (deeper = more powerful)
        """
        super(OthelloPolicyNetwork, self).__init__()
        
        # ============================================
        # PART 1: Initial Convolution
        # ============================================
        # This layer looks at the board and extracts basic patterns
        # Think: "Where are pieces? Are there groups? Edges?"
        
        self.conv_input = nn.Conv2d(
            in_channels=1,        # Input: 1 channel (the board)
            out_channels=num_filters,  # Output: many feature maps
            kernel_size=3,        # Look at 3x3 neighborhoods
            padding=1             # Keep size 8x8
        )
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # ============================================
        # PART 2: Residual Blocks (The "Thinking" Part)
        # ============================================
        # These blocks process the board deeply
        # Each block adds more sophisticated pattern recognition
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) 
            for _ in range(num_residual_blocks)
        ])
        
        # ============================================
        # PART 3: Policy Head (Decides Which Move)
        # ============================================
        # This part outputs: "How good is each of the 64 squares?"
        
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)
        
        # ============================================
        # PART 4: Value Head (Predicts Who Wins)
        # ============================================
        # This part outputs: "Who's winning this game?"
        
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Tensor of shape (batch_size, 1, 8, 8)
               - batch_size: number of board positions
               - 1: single channel (the board)
               - 8x8: board dimensions
        
        Returns:
            policy: Tensor of shape (batch_size, 64) - probabilities for each square
            value: Tensor of shape (batch_size, 1) - win probability
        """
        
        # ============================================
        # Step 1: Initial feature extraction
        # ============================================
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # ============================================
        # Step 2: Deep processing through residual blocks
        # ============================================
        for block in self.residual_blocks:
            x = block(x)
        
        # ============================================
        # Step 3: Compute policy (which move to play)
        # ============================================
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # Convert to log probabilities
        
        # ============================================
        # Step 4: Compute value (who's winning)
        # ============================================
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and +1
        
        return policy, value


class ResidualBlock(nn.Module):
    """
    Residual Block - A building block that helps networks learn better.
    
    Key idea: Instead of learning f(x), learn f(x) + x
    This "residual connection" makes training much easier for deep networks.
    
    This is the same idea used in ResNet (hence the name).
    """
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        The key line is: return F.relu(out + x)
        We add the input (x) back to the output (out)
        This is the "skip connection" or "residual connection"
        """
        residual = x  # Save the input
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # Add input back (the "residual" part!)
        out = F.relu(out)
        
        return out


# ============================================
# Helper Functions
# ============================================

def board_to_tensor(board, current_player):
    """
    Convert Othello board to neural network input tensor.
    
    Args:
        board: numpy array (8, 8) with values:
               1 = black piece
              -1 = white piece
               0 = empty
        current_player: 1 (black) or -1 (white)
    
    Returns:
        tensor: PyTorch tensor of shape (1, 1, 8, 8)
                From current player's perspective (always sees self as 1)
    """
    # Flip perspective so current player always sees themselves as 1
    board_from_player_perspective = board * current_player
    
    # Convert to PyTorch tensor and add batch + channel dimensions
    tensor = torch.FloatTensor(board_from_player_perspective).unsqueeze(0).unsqueeze(0)
    
    return tensor


def tensor_to_board(tensor):
    """
    Convert tensor back to board (for debugging).
    
    Args:
        tensor: PyTorch tensor of shape (1, 1, 8, 8)
    
    Returns:
        board: numpy array (8, 8)
    """
    return tensor.squeeze().numpy()


# ============================================
# Testing & Demonstration
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("Policy Network Demo")
    print("="*50)
    
    # Create network
    print("\n1. Creating network...")
    net = OthelloPolicyNetwork(num_filters=64, num_residual_blocks=2)
    
    # Count parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"   Network has {num_params:,} trainable parameters")
    
    # Create example board (starting position)
    print("\n2. Creating example board (starting position)...")
    board = np.zeros((8, 8))
    board[3][3] = -1  # white
    board[3][4] = 1   # black
    board[4][3] = 1   # black
    board[4][4] = -1  # white
    
    print("   Board:")
    print("   ", board)
    
    # Convert to tensor
    print("\n3. Converting board to tensor...")
    tensor = board_to_tensor(board, current_player=1)
    print(f"   Tensor shape: {tensor.shape}")
    
    # Forward pass (get predictions)
    print("\n4. Running forward pass (getting predictions)...")
    policy, value = net(tensor)
    
    print(f"   Policy shape: {policy.shape}")
    print(f"   Value shape: {value.shape}")
    
    # Show some predictions
    print("\n5. Policy predictions (first 10 squares):")
    policy_probs = torch.exp(policy[0])  # Convert log probs to probs
    for i in range(10):
        print(f"   Square {i}: {policy_probs[i].item():.6f}")
    
    print(f"\n6. Sum of all probabilities: {policy_probs.sum().item():.6f}")
    print(f"   (Should be 1.0)")
    
    print(f"\n7. Value prediction: {value.item():.4f}")
    print(f"   (Negative = white winning, Positive = black winning)")
    print(f"   (Close to 0 = draw)")
    
    print("\n" + "="*50)
    print("Network created successfully!")
    print("="*50)
    
    print("\nNext steps:")
    print("- This network is UNTRAINED (random predictions)")
    print("- We need to train it through self-play")
    print("- Then it will learn good moves!")