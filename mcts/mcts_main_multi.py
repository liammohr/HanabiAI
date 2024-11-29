import os
import psutil
from multiprocessing import Pool, cpu_count
from game_state import GameState
from mcts import RISMCTS
from mcts_node import HanabiNode
import numpy as np
import time  # For measuring runtime

PRINTS = False
RUNS = 99  # Number of games to run


def set_high_priority():
    """
    Sets the current process priority to high.
    Works on Windows, Linux, and macOS.
    """
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt':  # Windows
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Unix/Linux/Mac
            p.nice(-10)  # Negative values mean higher priority
    except Exception as e:
        print(f"Could not set high priority: {e}")


def run_single_game(time_limit):
    """
    Runs a single game and returns the final score.
    """
    set_high_priority()  # Ensure this process runs with high priority

    game = GameState(0)
    player = 0
    while not game.game_ended()[0]:
        new_state = game.__deepcopy__()
        new_state.root = player
        agent = RISMCTS(HanabiNode(new_state, player=player), time_limit)
        action = agent.run()
        if action.action_type == "play":
            game.play_card(action.player, action.card_idx)
        elif action.action_type == "discard":
            game.discard_card(action.player, action.card_idx)
        elif action.action_type == "hint":
            game.give_hint(action.destination, action.hint_type, action.hint_value)
        if PRINTS:
            print(action)
            print(game.board)
            print(game.hands)
        player = (player + 1) % 2
    return game.game_ended()[1]


def run_games_parallel(time_limit):
    """
    Runs multiple games in parallel and collects their scores.
    """
    num_cores = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} cores for parallel processing.")

    start_time = time.time()  # Start timing

    with Pool(processes=num_cores) as pool:
        scores = pool.map(run_single_game, [time_limit] * RUNS)

    end_time = time.time()  # End timing
    total_runtime = end_time - start_time

    average = np.mean(scores)
    minimum = np.min(scores)
    maximum = np.max(scores)
    std_dev = np.std(scores)
    median = np.median(scores)
    summary = (
        f"Scores Statistics:\n"
        f"-------------------\n"
        f"Average: {average:.2f}\n"
        f"Minimum: {minimum}\n"
        f"Maximum: {maximum}\n"
        f"Standard Deviation: {std_dev:.2f}\n"
        f"Median: {median:.2f}\n"
        f"Number of Scores: {len(scores)}\n"
        f"Total Runtime: {total_runtime:.2f} seconds\n"
        f"Average Time per Game: {total_runtime / RUNS:.2f} seconds"
    )

    # Print statistics
    print(summary)

    # Save scores and summary to a file
    with open("mcts_5sec_scores.txt", "w") as file:
        file.write("Scores:\n")
        file.write(", ".join(map(str, scores)) + "\n\n")
        file.write(summary)


if __name__ == '__main__':
    set_high_priority()  # Set high priority for the main process
    run_games_parallel(5)  # Run with a 2-second time limit per game
