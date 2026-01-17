"""
AlphaZero Training Loop for Othello

This connects everything:
1. Self-play generates games using MCTS
2. Training updates the network from self-play data
3. Repeat to improve
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from othello import OthelloGame
from policy_network import OthelloPolicyNetwork, board_to_tensor
from mcts import MCTS


class AlphaZeroTrainer:
    """
    Main training loop for AlphaZero.
    """
    
    def __init__(self, network, game, num_simulations=50, c_puct=1.0):
        self.network = network
        self.game = game
        self.mcts = MCTS(network, game, num_simulations, c_puct)
        
        # Training hyperparameters
        self.batch_size = 32
        self.replay_buffer = deque(maxlen=10000)  # Store recent games
        self.optimizer = optim.Adam(network.parameters(), lr=0.001)
        
    def self_play_game(self):
        """
        Play one complete game using MCTS.
        Returns training examples: (board, mcts_policy, outcome)
        """
        game = OthelloGame()
        current_player = 1
        
        training_examples = []
        
        while not game.is_game_over():
            # Get MCTS policy for current position
            board_copy = game.board.copy()
            mcts_policy = self.mcts.search(board_copy, current_player)
            
            # Store training example
            training_examples.append((
                board_copy.copy(),
                current_player,
                mcts_policy
            ))
            
            # Sample move from MCTS policy (with temperature for exploration)
            legal_moves = game.get_legal_moves(current_player)
            
            if len(legal_moves) == 0:
                current_player = -current_player
                continue
            
            # Choose move (stochastic during training)
            legal_policy = np.zeros(64)
            for move in legal_moves:
                move_idx = move[0] * 8 + move[1]
                legal_policy[move_idx] = mcts_policy[move_idx]
            
            if legal_policy.sum() > 0:
                legal_policy = legal_policy / legal_policy.sum()
                move_idx = np.random.choice(64, p=legal_policy)
                move = (move_idx // 8, move_idx % 8)
            else:
                move = random.choice(legal_moves)
            
            # Make move
            game.make_move(move[0], move[1], current_player)
            current_player = -current_player
        
        # Get final outcome
        winner = game.get_winner()
        
        # Augment training examples with final outcome
        augmented_examples = []
        for board, player, policy in training_examples:
            # Value from perspective of player who made the move
            value = winner * player
            augmented_examples.append((board, player, policy, value))
        
        return augmented_examples
    
    def train_network(self):
        """
        Train network on batch from replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        boards, players, policies, values = zip(*batch)
        
        # Convert to tensors
        board_tensors = torch.stack([
            board_to_tensor(board, player) 
            for board, player in zip(boards, players)
        ]).squeeze(1)
        
        target_policies = torch.FloatTensor(policies)
        target_values = torch.FloatTensor(values).unsqueeze(1)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred_policies, pred_values = self.network(board_tensors)
        
        # Loss functions
        policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
        value_loss = torch.mean((target_values - pred_values) ** 2)
        
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train(self, num_iterations=100, games_per_iteration=10):
        """
        Main training loop.
        """
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # Self-play phase
            print("Self-play...")
            for game_num in range(games_per_iteration):
                examples = self.self_play_game()
                self.replay_buffer.extend(examples)
                print(f"  Game {game_num + 1}/{games_per_iteration}: {len(examples)} positions")
            
            # Training phase
            print("\nTraining...")
            num_train_steps = len(self.replay_buffer) // self.batch_size
            
            for step in range(min(num_train_steps, 50)):  # Cap training steps
                losses = self.train_network()
                
                if losses and step % 10 == 0:
                    print(f"  Step {step}: Loss={losses['total_loss']:.4f} "
                          f"(Policy={losses['policy_loss']:.4f}, Value={losses['value_loss']:.4f})")
            
            print(f"\nReplay buffer size: {len(self.replay_buffer)}")


def main():
    """
    Run AlphaZero training.
    """
    print("Initializing AlphaZero for Othello...")
    
    # Create network
    network = OthelloPolicyNetwork(num_filters=64, num_residual_blocks=2)
    
    # Create game
    game = OthelloGame()
    
    # Create trainer
    trainer = AlphaZeroTrainer(
        network=network,
        game=game,
        num_simulations=25,  # Reduced for speed
        c_puct=1.0
    )
    
    print("Starting training...")
    print("Note: This is a minimal implementation for learning purposes.")
    print("Full AlphaZero would need weeks of training on GPUs.\n")
    
    # Train for a few iterations
    trainer.train(num_iterations=5, games_per_iteration=3)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print("\nWhat we built:")
    print("✓ Complete Othello game environment")
    print("✓ Policy network with residual blocks")
    print("✓ Monte Carlo Tree Search")
    print("✓ Self-play data generation")
    print("✓ Training loop with replay buffer")
    print("\nThis is a complete (minimal) AlphaZero implementation!")
    print("The concepts scale to larger games and models.")
    
    # Save the trained network
    torch.save(network.state_dict(), 'alphazero_othello.pt')
    print("\nNetwork saved to 'alphazero_othello.pt'")


if __name__ == "__main__":
    main()