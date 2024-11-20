from game_state import GameState
from mcts import RISMCTS
from mcts_node import HanabiNode


def run_game(time_limit):
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
        print(action)
        print(game.board)
        print(game.hands)
        player = (player + 1) % 2
    print(game.game_ended())


if __name__ == '__main__':
    run_game(1)
