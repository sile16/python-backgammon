import time
import numpy as np
from tqdm import tqdm
from state import State, set_debug, set_random_seed, MoveSequence

def initialize_stats():
    """Initialize statistics counters with default values"""
    return {
        'winner_counts': {0: 0, 1: 0},  # Using 0 for WHITE, 1 for BLACK
        'points_counts': {0: 0, 1: 0},
        'win_types': {
            'single': 0,      # 1 point
            'gammon': 0,      # 2 points
            'backgammon': 0   # 3 points
        },
        'max_moves_seen': 0
    }

def update_statistics(stats, winner, points, max_moves):
    """Update game statistics safely"""
    if winner not in (0, 1):  # WHITE = 0, BLACK = 1
        raise ValueError(f"Invalid winner value: {winner}")
    
    stats['winner_counts'][winner] += 1
    stats['points_counts'][winner] += points
    stats['max_moves_seen'] = max(stats['max_moves_seen'], max_moves)
    
    # Update win type counts
    if points == 1:
        stats['win_types']['single'] += 1
    elif points == 2:
        stats['win_types']['gammon'] += 1
    elif points == 3:
        stats['win_types']['backgammon'] += 1
    else:
        raise ValueError(f"Invalid points value: {points}")

def print_summary_statistics(stats, duration, n_games):
    """Print summary statistics"""
    print("\nGame Results Summary:")
    print("-" * 50)
    print(f"Total Games Played: {n_games}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speed: {n_games/duration:.2f} games/second")
    print("\nWin Statistics:")
    print(f"White (Player 0) wins: {stats['winner_counts'][0]} ({stats['winner_counts'][0]/n_games*100:.1f}%)")
    print(f"Black (Player 1) wins: {stats['winner_counts'][1]} ({stats['winner_counts'][1]/n_games*100:.1f}%)")
    print("\nPoints Statistics:")
    print(f"White total points: {stats['points_counts'][0]}")
    print(f"Black total points: {stats['points_counts'][1]}")
    print("\nWin Types:")
    print(f"Single wins: {stats['win_types']['single']} ({stats['win_types']['single']/n_games*100:.1f}%)")
    print(f"Gammon wins: {stats['win_types']['gammon']} ({stats['win_types']['gammon']/n_games*100:.1f}%)")
    print(f"Backgammon wins: {stats['win_types']['backgammon']} ({stats['win_types']['backgammon']/n_games*100:.1f}%)")
    print(f"\nMaximum legal moves in a single turn: {stats['max_moves_seen']}")

def play_single_game(stats):
    """Play a single game and update statistics"""
    state = State()
    state.pick_first_player()
    set_debug(False)
    
    max_moves = 0
    
    while not state.isTerminal():
        try:
            state.roll_dice()
            moves = state.get_legal_moves()
            mlen = len(moves)
            
            max_moves = max(max_moves, mlen)
            
            if mlen > 0:
                move_index = np.random.randint(mlen)
                if move_index < len(moves):
                    state.do_moves(moves[move_index])
            
            if state.isTerminal():
                update_statistics(stats, state.winner, state.points, max_moves)
                break
                
        except Exception as e:
            print(f"\nError during game: {str(e)}")
            return False
    
    return True

def run_games(n_games):
    """Run multiple games with progress bar and collect statistics"""
    start_time = time.time()
    stats = initialize_stats()
    completed_games = 0
    
    try:
        # Setup progress bar
        with tqdm(total=n_games, desc="Playing games", unit="game") as pbar:
            for _ in range(n_games):
                if play_single_game(stats):
                    completed_games += 1
                pbar.update(1)
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        duration = time.time() - start_time
        if completed_games > 0:
            print_summary_statistics(stats, duration, completed_games)
        else:
            print("\nNo games completed successfully")

if __name__ == "__main__":
    N_GAMES = 100
    run_games(N_GAMES)