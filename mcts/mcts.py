import copy
import random
import time

import numpy as np

from mcts_node import HanabiNode, MCTSNode
from game_state import Action


class RISMCTS:
    """
    Represents the RIS-MCTS algorithm for the Hanabi game.
    """

    def __init__(self, root: HanabiNode, time_limit: float):
        """
        Initialize RIS-MCTS.

        Args:
            root: The root node of the search tree.
            time_limit: The time limit for the algorithm (in seconds).
        """
        self.root: HanabiNode = root
        self.time_limit = time_limit
        self.saved_hand = None  # To save the hand for the root player during redeterminization

    def run(self) -> Action:
        """
        Run the RIS-MCTS algorithm.

        Returns:
            Action: The best action from the root node after the search.
        """
        end_time = time.time() + self.time_limit
        while time.time() < end_time:
            # Step 3: Redeterminize the root hand
            player = self.root.state.root
            self.root.state.redeterminize_hand(player,bypass=True)
            node = self.root
            while node.is_fully_expanded() and not node.leaf:
                new_node = node.select_child()
                self.exit_node(node, player)
                player = new_node.player
                node = new_node
                self.enter_node(node, player)
            node = self.expand(node)
            score = node.rollout()
            node.backpropagate(score)
        best = self.root.select_child()
        # print(self.root.visits)
        return best.action



    def enter_node(self, node:MCTSNode, player):
        if self.root.state.root == player:
            return

        self.saved_hand = copy.deepcopy(node.state.hands[player])

        node.state.redeterminize_hand(player)

    def exit_node(self, node:MCTSNode, player):

        if self.saved_hand is not None:
            node.state.restore_hand(player, self.saved_hand, self.root.state.root)

    def expand(self, node:HanabiNode):
        if node.state.game_ended()[0]:
            return node
        actions = node.get_legal_actions()
        children = [c.action for c in node.children]

        return node.expand(random.choice([ac for ac in actions if ac not in children]))

