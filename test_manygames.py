import time
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager
import os
import sys

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

def merge_stats(stats_list):
    """Merge statistics from multiple processes"""
    merged = initialize_stats()
    
    for stats in stats_list:
        for winner in (0, 1):
            merged['winner_counts'][winner] += stats['winner_counts'][winner]
            merged['points_counts'][winner] += stats['points_counts'][winner]
        
        for win_type in merged['win_types']:
            merged['win_types'][win_type] += stats['win_types'][win_type]
        
        merged['max_moves_seen'] = max(merged['max_moves_seen'], 
                                     stats['max_moves_seen'])
    
    return merged

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

def play_single_game():
    """Play a single game and return statistics"""
    stats = initialize_stats()
    state = BGGame()
    state.pick_first_player()
    set_debug(False)
    
    max_moves = 0
    no_moves_found = 0
    
    while not state.isTerminal():
        try:
            state.roll_dice()
            moves = state.get_legal_moves2()
            mlen = len(moves)
            
            max_moves = max(max_moves, mlen)
            move_index = np.random.randint(mlen)
            state.do_moves(moves[move_index])
            
           
            if state.isTerminal():
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
    

def worker(n_games, result_queue, process_id, progress_queue):
    """Worker process function"""
    np.random.seed(os.getpid() + process_id)  # Ensure different seeds for each process
    process_stats = initialize_stats()
    
    for _ in range(n_games):
        game_stats = play_single_game()
        if game_stats:
            # Merge this game's stats into process stats
            for winner in (0, 1):
                process_stats['winner_counts'][winner] += game_stats['winner_counts'][winner]
                process_stats['points_counts'][winner] += game_stats['points_counts'][winner]
            for win_type in process_stats['win_types']:
                process_stats['win_types'][win_type] += game_stats['win_types'][win_type]
            process_stats['max_moves_seen'] = max(process_stats['max_moves_seen'], 
                                                game_stats['max_moves_seen'])
        progress_queue.put(1)  # Report progress
    
    result_queue.put(process_stats)

def run_games(n_games):
    """Run multiple games using multiple processes"""
    start_time = time.time()
    n_cores = mp.cpu_count()
    games_per_process = n_games // n_cores
    remaining_games = n_games % n_cores
    
    print(f"Running {n_games} games using {n_cores} CPU cores")
    
    with Manager() as manager:
        result_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        processes = []
        
        # Start progress bar process
        with tqdm(total=n_games, desc="Playing games", unit="game") as pbar:
            # Start worker processes
            for i in range(n_cores):
                n_games_for_this_process = games_per_process + (1 if i < remaining_games else 0)
                p = mp.Process(target=worker, 
                             args=(n_games_for_this_process, result_queue, i, progress_queue))
                processes.append(p)
                p.start()
            
            # Update progress bar
            completed = 0
            while completed < n_games:
                progress_queue.get()
                completed += 1
                pbar.update(1)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        stats_list = []
        while not result_queue.empty():
            stats_list.append(result_queue.get())
        
        # Merge results from all processes
        final_stats = merge_stats(stats_list)
        
        duration = time.time() - start_time
        print_summary_statistics(final_stats, duration, n_games)

if __name__ == "__main__":
    N_GAMES = 50000  # Increased number of games to better utilize multiple cores
    run_games(N_GAMES)