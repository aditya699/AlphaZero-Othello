# AlphaZero for Othello - Complete Implementation

A minimal but complete implementation of AlphaZero for Othello, built for learning purposes.

## What This Is

This project implements the core AlphaZero algorithm:
1. **Neural Network** - Evaluates board positions and suggests moves
2. **Monte Carlo Tree Search (MCTS)** - Searches game tree using network guidance
3. **Self-Play** - Plays games against itself to generate training data
4. **Training Loop** - Improves network from self-play data

## Project Structure

```
├── othello.py          # Game environment (rules, legal moves, winner)
├── policy_network.py   # Neural network (policy + value heads)
├── mcts.py            # Monte Carlo Tree Search implementation
├── train.py           # Complete training loop (self-play + training)
└── README.md          # This file
```

## How It Works

### 1. Game Environment (`othello.py`)
- 8x8 board with standard Othello rules
- Tracks legal moves, flipping pieces, game termination
- Returns winner at end of game

### 2. Neural Network (`policy_network.py`)
- **Input**: 8x8 board state
- **Architecture**: Convolutional layers + residual blocks (like ResNet)
- **Outputs**:
  - Policy: 64 probabilities (which square to play)
  - Value: Win probability (-1 to +1)

### 3. MCTS (`mcts.py`)
Four phases repeated many times:
1. **Selection**: Walk down tree using UCB formula
2. **Expansion**: Add new node when reaching leaf
3. **Evaluation**: Use neural network to evaluate position
4. **Backpropagation**: Update all nodes in path with result

### 4. Training Loop (`train.py`)
1. **Self-play**: Network plays against itself using MCTS
2. **Store data**: Save (board, MCTS policy, game outcome)
3. **Train**: Update network to match MCTS policies and predict outcomes
4. **Repeat**: Improved network → better MCTS → better training data

## Key Concepts Learned

### Reinforcement Learning
- Value function: "How good is this position?"
- Policy: "Which action should I take?"
- Self-improvement through self-play

### Deep Learning
- Convolutional networks for spatial patterns
- Residual connections for deep networks
- Multi-task learning (policy + value heads)

### Search Algorithms
- Monte Carlo Tree Search
- UCB (Upper Confidence Bound) exploration
- Balancing exploitation vs exploration

### Training Dynamics
- Replay buffer for stable training
- Policy distillation from MCTS to network
- Bootstrap learning (network improves itself)

## Running the Code

```bash
# Train the network
python train.py

# Play a random game
python othello.py

# Test the network
python policy_network.py
```

## Why This Matters

This is the same algorithm that:
- Beat the world champion at Go (AlphaGo)
- Mastered chess from scratch in 4 hours (AlphaZero)
- Discovered new strategies in dozens of games

The concepts scale:
- Bigger networks → more capacity
- More simulations → better search
- More training → superhuman play

## What We Didn't Implement (for production)

- Data augmentation (rotations, flips)
- Parallel self-play workers
- GPU acceleration
- Dirichlet noise for exploration
- Evaluation against older versions
- Temperature decay schedule
- Proper hyperparameter tuning

## From Learning to Production

To make this production-ready:
1. Add GPU support (`model.cuda()`)
2. Parallelize self-play (multiprocessing)
3. Add data augmentation
4. Train for 1000+ iterations
5. Implement evaluation pipeline
6. Save checkpoints regularly

## Next Steps for Learning

Now that you understand AlphaZero, explore:
- **MuZero**: Learns the game rules too
- **AlphaTensor**: Discovers algorithms (matrix multiplication)
- **LLM Pre-training**: Next-token prediction at scale
- **RLHF**: Aligning LLMs with human preferences

## Key Takeaway

AlphaZero combines:
- **Deep learning** (pattern recognition)
- **Tree search** (planning)
- **Reinforcement learning** (self-improvement)

This same combination powers modern AI systems, from game-playing to reasoning models.

---

**Built for learning. Concepts scale to production.**