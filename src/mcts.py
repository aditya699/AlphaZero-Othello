"""
Monte Carlo Tree Search for Othello AlphaZero
"""

import numpy as np
import math
import torch


class MCTSNode:
    """
    Node in the MCTS search tree.
    """
    
    def __init__(self, board, current_player, parent=None, prior_prob=0):
        self.board = board
        self.current_player = current_player
        self.parent = parent
        self.prior_prob = prior_prob
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = self._ucb_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def _ucb_score(self, child, c_puct):
        prior_score = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
        value_score = child.value()
        return value_score + prior_score
    
    def expand(self, legal_moves, policy_probs):
        for move in legal_moves:
            if move not in self.children:
                move_idx = move[0] * 8 + move[1]
                prior = policy_probs[move_idx]
                self.children[move] = MCTSNode(
                    board=None,
                    current_player=-self.current_player,
                    parent=self,
                    prior_prob=prior
                )
    
    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
        
    def update_recursive(self, value):
        self.update(value)
        if self.parent is not None:
            self.parent.update_recursive(-value)


class MCTS:
    """
    Monte Carlo Tree Search.
    """
    
    def __init__(self, network, game, num_simulations=100, c_puct=1.0):
        self.network = network
        self.game = game
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, board, current_player):
        root = MCTSNode(board, current_player)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion & Evaluation
            value = self._expand_and_evaluate(node, search_path)
            
            # Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value
                
        return self._get_policy(root)
    
    def _expand_and_evaluate(self, node, search_path):
        parent = search_path[-2] if len(search_path) > 1 else None
        
        if parent is None:
            board = node.board
        else:
            parent_node = search_path[-2]
            board = self._apply_move(parent_node.board, parent_node.current_player, self._get_action(parent_node, node))
        
        node.board = board
        
        legal_moves = self.game.get_legal_moves(board, node.current_player)
        
        if len(legal_moves) == 0:
            if self.game.is_game_over(board):
                winner = self.game.get_winner(board)
                return winner * node.current_player
            else:
                return 0
        
        policy, value = self._evaluate(board, node.current_player)
        node.expand(legal_moves, policy)
        
        return value
    
    def _evaluate(self, board, current_player):
        from policy_network import board_to_tensor
        
        tensor = board_to_tensor(board, current_player)
        
        with torch.no_grad():
            policy_logits, value = self.network(tensor)
        
        policy_probs = torch.exp(policy_logits).squeeze().numpy()
        value = value.item()
        
        return policy_probs, value
    
    def _apply_move(self, board, player, move):
        new_board = board.copy()
        self.game.make_move(new_board, move, player)
        return new_board
    
    def _get_action(self, parent, child):
        for action, c in parent.children.items():
            if c is child:
                return action
        return None
    
    def _get_policy(self, root):
        policy = np.zeros(64)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for action, child in root.children.items():
            move_idx = action[0] * 8 + action[1]
            policy[move_idx] = child.visit_count / total_visits if total_visits > 0 else 0
            
        return policy


# Demo
if __name__ == "__main__":
    from policy_network import OthelloPolicyNetwork, board_to_tensor
    import sys
    sys.path.append('.')
    
    # You'll need to import your game environment
    # from othello_game import OthelloGame
    
    print("MCTS implementation ready!")
    print("Integrate with your Othello game environment to test.")