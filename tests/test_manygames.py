import time
import numpy as np
from tqdm import tqdm
import os
import sys

# Current best Cython single thread is 120 games / second, 708 with 8 threads

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the build/lib directory to Python's path
build_lib_dir = os.path.join(project_root, 'build', 'lib')
if build_lib_dir not in sys.path:
    sys.path.insert(0, build_lib_dir)

from bg_game import BGGame, set_debug

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
    """Update game statistics"""
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

def play_single_game():
    """Play a single game and return statistics"""
    stats = initialize_stats()
    state = BGGame()
    state.set_player(0)
    state.randomize_seed()
    state.roll_dice()
    
    set_debug(False)
    
    max_moves = 0
    
    while not state.isTerminal():
        try:
            moves = state.legal_actions()
            max_moves = max(max_moves, len(moves))
            move_index = np.random.randint(len(moves))
            obs, reward, done = state.step(moves[move_index])
           
            if done:
                update_statistics(stats, state.winner, state.points, max_moves)
                return stats
                
        except Exception as e:
            print(f"\nError during game: {str(e)}")
            return None
    
    return stats

def print_state(state):
    for ms in state.move_seq_list:
        print(f"Dice {ms.dice[0]} {ms.dice[1]}")
        print(f"Board state:\n{ms}")
        print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
        print(np.array_str(ms.final_board, precision=2, suppress_small=True))
    
    print(f"Current Roll: {state.dice[0]} {state.dice[1]}")
    print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
    print(np.array_str(state.board, precision=2, suppress_small=True))

def run_games(n_games):
    """Run multiple games sequentially"""
    start_time = time.time()
    print(f"Running {n_games} games on a single thread")
    
    stats = initialize_stats()
    
    with tqdm(total=n_games, desc="Playing games", unit="game") as pbar:
        for _ in range(n_games):
            game_stats = play_single_game()
            if game_stats:
                # Merge this game's stats into the main stats
                for winner in (0, 1):
                    stats['winner_counts'][winner] += game_stats['winner_counts'][winner]
                    stats['points_counts'][winner] += game_stats['points_counts'][winner]
                for win_type in stats['win_types']:
                    stats['win_types'][win_type] += game_stats['win_types'][win_type]
                stats['max_moves_seen'] = max(stats['max_moves_seen'], 
                                             game_stats['max_moves_seen'])
            pbar.update(1)
    
    duration = time.time() - start_time
    print_summary_statistics(stats, duration, n_games)

if __name__ == "__main__":
    N_GAMES = 10000  # You can adjust the number of games as needed
    #run_games(N_GAMES)
    bg = BGGame()
    bg.play_n_games_to_end(N_GAMES)
