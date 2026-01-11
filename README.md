# AlphaZero Othello: Learning RL + Deep Learning

Implementation of AlphaZero for Othello to understand reinforcement learning and its connection to modern LLM post-training.

---

## Motivation

Started from trying to understand:
- How RLHF works in practice
- Why Karpathy says "RL is terrible" 
- What o1's "test-time compute" actually means
- How systems learn without labeled data

Building AlphaZero from scratch to understand these concepts concretely, not just theoretically.

---

## What We're Learning

### Core RL Concepts
- Monte Carlo Tree Search (selection, expansion, simulation, backprop)
- Self-play for data generation
- Policy improvement through search
- Credit assignment problem
- Exploration vs exploitation (UCB formula)

### Deep Learning Components
- CNNs for spatial data (translation equivariance)
- Residual blocks for deep networks
- Multi-head outputs (policy + value)
- Training on self-generated data

### The Hard Problems
- Sparse rewards (outcome only known at game end)
- Noisy gradients (upweighting entire winning trajectories)
- Sample efficiency (thousands of games needed)
- Model collapse (training on own outputs)

---

## Technical Architecture

### Network
```
Input: 8×8 board (1 channel)
  ↓
Conv2d(1→128, 3×3) + BatchNorm + ReLU
  ↓
4× ResidualBlock(128→128)
  ↓
  ├→ Policy Head → 64 logits (move distribution)
  └→ Value Head → 1 output ∈ [-1,+1] (position eval)
```

### MCTS Algorithm
```python
for simulation in range(num_simulations):
    # 1. Selection: traverse tree using UCB
    node = select_leaf(root)
    
    # 2. Expansion: add children for legal moves
    expand(node, network_policy)
    
    # 3. Evaluation: network predicts value
    value = network.forward(node.state)
    
    # 4. Backup: propagate value up tree
    backpropagate(node, value)

# Return visit counts as improved policy
return visit_distribution(root)
```

### Training Loop
```python
for iteration in range(num_iterations):
    # Generate data via self-play
    games = play_games(network, mcts, n=100)
    
    # Train on MCTS-improved policies
    for batch in games:
        policy_loss = -sum(mcts_policy * log(network_policy))
        value_loss = (network_value - game_result)²
        loss = policy_loss + value_loss
        optimize(loss)
```

---

## Why RL is Hard (Karpathy's Perspective)

From Dwarkesh podcast:

**The Problem**:
```
Generate 100 solution attempts → 3 succeed, 97 fail

RL upweights every token in successful attempts,
even the wrong steps that happened to lead to success.

"You're sucking supervision through a straw" - one bit 
(win/loss) used to update entire trajectory.
```

**High Variance**: Assumes every step in winning trajectory was correct.

**Credit Assignment**: Can't tell which moves actually mattered.

**AlphaZero's Workaround**: MCTS provides richer signal than just win/loss.

---

## Connection to Modern LLMs

| AlphaZero (Games) | LLM Post-Training |
|-------------------|-------------------|
| MCTS explores move tree | Search over reasoning chains |
| Network learns from MCTS | Model learns from verified solutions |
| Self-play generates data | Model generates training data |
| Verification = win/loss | Verification = passes tests |

**o1-style models**:
- Use test-time compute (multiple reasoning attempts)
- Verify solutions (like checking game outcome)
- Train on verified trajectories (RLVR)

**Open problems** (same as AlphaZero faced):
- Model collapse when training on own outputs
- Process supervision (reward each step, not just outcome)
- LLM judges are gameable (adversarial examples)

---

## Implementation Details

### Files
```
src/
├── othello.py          # Game rules, legal moves, win detection
├── policy_network.py   # CNN: Conv layers + ResBlocks → policy/value
├── mcts.py            # Tree search: UCB selection + backprop
└── self_play.py       # Generate training data via self-play
```

### Key Design Decisions

**Why CNNs?**
- Translation equivariance: pattern detector works anywhere on board
- Much fewer parameters than fully connected

**Why Residual Blocks?**
- Enable deep networks (4+ layers)
- Gradients flow through skip connections

**Why MCTS + Network?**
- Network alone: fast but weak
- MCTS alone: strong but slow
- Combined: network learns to play at MCTS strength without search cost

**Why Self-Play?**
- No human data needed
- Natural curriculum (always play at your level)
- Unlimited training data

---

## Current Status

**Working**:
- ✓ Othello environment (board, moves, game over detection)
- ✓ Policy network (tested forward pass)
- ✓ MCTS implementation (4 phases)
- ✓ Self-play data generation

**TODO**:
- Training script (optimize network on self-play data)
- Full training loop (iterate self-play → train)
- Evaluation metrics (Elo rating, win rate vs baselines)
- Ablation studies (network size, MCTS simulations, etc.)

---

## Technical Challenges Encountered

**Understanding MCTS**: 
- Confused "backpropagation" in MCTS (updating stats) with neural network backprop (gradients)
- Took multiple sessions to understand when network is called vs random rollouts

**Tree Growth**:
- Understanding asymmetric tree (explores promising moves more)
- How statistics accumulate over simulations

**Self-Play Mechanics**:
- When to discard tree (after each real move)
- Temperature scheduling (exploration early, exploitation late)
- How to assign results to alternating players

**Network Training**:
- Why KL divergence for policy (not just cross-entropy)
- Balancing policy loss vs value loss
- Preventing overfitting on self-play data

---

## Learning Resources

**Papers**:
- Silver et al. 2017 - "Mastering the game of Go without human knowledge"
- Silver et al. 2018 - "A general reinforcement learning algorithm that masters chess, shogi, and Go"

**Concepts**:
- UCB formula (Auer et al. 2002)
- ResNet (He et al. 2015)
- Monte Carlo methods (Ulam 1940s, Manhattan Project)

**Discussions**:
- Karpathy/Dwarkesh podcast on RL limitations
- Why process supervision > outcome supervision
- Model collapse in self-training

---

## Open Questions

1. Minimum compute requirements for convergence?
2. Does network architecture matter much? (filters, depth)
3. How many iterations until strategies emerge?
4. Can we visualize what network learns?
5. Sample efficiency vs AlphaGo Zero (they used way more compute)

---

## Notes

This is a learning project. Code prioritizes clarity over efficiency. Many design choices are pedagogical (understanding why things work) rather than optimal (best performance).

The goal: build intuition for RL + DL by implementing from scratch, making mistakes, and understanding why solutions work.

---

## License

MIT

---

*Learning by building. Understanding by implementing.*