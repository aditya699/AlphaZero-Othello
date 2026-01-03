import numpy as np

class OthelloGame:
    def __init__(self):
        self.board_size = 8
        self.board = np.zeros((8, 8), dtype=int)
        # Initial position: center 4 squares
        # 1 = black, -1 = white, 0 = empty
        self.board[3][3] = -1  # white
        self.board[3][4] = 1   # black
        self.board[4][3] = 1   # black
        self.board[4][4] = -1  # white
        
    def display(self):
        """Print the board"""
        print("  0 1 2 3 4 5 6 7")
        for i in range(8):
            print(f"{i}", end=" ")
            for j in range(8):
                if self.board[i][j] == 1:
                    print("●", end=" ")  # black
                elif self.board[i][j] == -1:
                    print("○", end=" ")  # white
                else:
                    print("·", end=" ")  # empty
            print()

    def get_legal_moves(self, player):
        """
        Get all legal moves for a player
        player: 1 (black) or -1 (white)
        Returns: list of (row, col) tuples
        """
        legal_moves = []
        
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    legal_moves.append((row, col))
        
        return legal_moves

    def is_valid_move(self, row, col, player):
        """
        Check if a move is valid
        Must flip at least one opponent piece
        """
        # Square must be empty
        if self.board[row][col] != 0:
            return False
        
        # Check all 8 directions
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, player):
                return True
        
        return False

    def _check_direction(self, row, col, dr, dc, player):
        """
        Check if placing a piece flips opponent pieces in this direction
        """
        r, c = row + dr, col + dc
        found_opponent = False
        
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == 0:  # Empty square
                return False
            elif self.board[r][c] == -player:  # Opponent piece
                found_opponent = True
            elif self.board[r][c] == player:  # Our piece
                return found_opponent  # Valid if we found opponent pieces in between
            
            r += dr
            c += dc
        
        return False
    def make_move(self, row, col, player):
        """
        Place a piece and flip opponent pieces
        Returns: True if move was successful, False otherwise
        """
        if not self.is_valid_move(row, col, player):
            return False
        
        # Place the piece
        self.board[row][col] = player
        
        # Flip pieces in all valid directions
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, player):
                self._flip_pieces(row, col, dr, dc, player)
        
        return True

    def _flip_pieces(self, row, col, dr, dc, player):
        """
        Flip opponent pieces in a direction
        """
        r, c = row + dr, col + dc
        
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == -player:  # Opponent piece
                self.board[r][c] = player     # Flip it!
            elif self.board[r][c] == player:  # Our piece
                break  # Stop flipping
            
            r += dr
            c += dc


    def is_game_over(self):
        """
        Game is over when neither player has legal moves
        """
        return len(self.get_legal_moves(1)) == 0 and len(self.get_legal_moves(-1)) == 0

    def get_winner(self):
        """
        Count pieces and determine winner
        Returns: 1 (black wins), -1 (white wins), 0 (draw)
        """
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0

    def get_score(self):
        """
        Returns: (black_count, white_count)
        """
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        return (black_count, white_count)

        


if __name__ == "__main__":
    import random
    
    game = OthelloGame()
    current_player = 1  # Black starts
    move_count = 0
    
    print("Playing a game with random moves...\n")
    game.display()
    
    while not game.is_game_over():
        legal_moves = game.get_legal_moves(current_player)
        
        if len(legal_moves) == 0:
            # No legal moves, pass turn
            print(f"\n{'Black' if current_player == 1 else 'White'} has no legal moves, passing...")
            current_player = -current_player
            continue
        
        # Pick random legal move
        move = random.choice(legal_moves)
        player_name = 'Black' if current_player == 1 else 'White'
        print(f"\n{player_name} plays at {move}")
        
        game.make_move(move[0], move[1], current_player)
        game.display()
        
        black_score, white_score = game.get_score()
        print(f"Score - Black: {black_score}, White: {white_score}")
        
        # Switch player
        current_player = -current_player
        move_count += 1
        
        if move_count > 60:  # Safety limit
            break
    
    # Game over
    print("\n" + "="*30)
    print("GAME OVER!")
    black_score, white_score = game.get_score()
    print(f"Final Score - Black: {black_score}, White: {white_score}")
    
    winner = game.get_winner()
    if winner == 1:
        print("Black wins! ●")
    elif winner == -1:
        print("White wins! ○")
    else:
        print("It's a draw!")