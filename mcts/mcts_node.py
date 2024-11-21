import copy
from typing import List, Optional
import math
import random

import numpy as np

from game_state import Action, MAX_HINTS, GameState
from rules import Rules


class MCTSNode:
    """
    A base MCTS Node class for general Monte Carlo Tree Search.
    """

    def __init__(self, state: GameState, action: Action = None, parent: Optional["MCTSNode"] = None,
                 player=None) -> None:
        """
        Initialize an MCTS node.

        Args:
            state: The game state associated with this node.
            parent: The parent node of this node (default is None).
        """
        self.state: GameState = state
        self.player = player
        if player is None:
            self.player = state.root
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.leaf = False

    def backpropagate(self, reward: float) -> None:
        """
        Backpropagate the reward through the tree.

        Args:
            reward: The reward to propagate.
        """
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def select_child(self):
        """
        Select the best direct child node using UCB1.

        Returns:
            MCTSNode: The selected child node.
        """

        def ucb1(node):
            if node.visits == 0:
                return float("inf")
            return (node.value / node.visits) + 0.3 * math.sqrt(
                math.log(self.visits) / node.visits
            )
        if len(self.children) == 0:
            self.leaf = True
            return self
        return max(self.children, key=ucb1)

    def is_fully_expanded(self) -> bool:
        """
        Check if the node is fully expanded (all possible actions explored).

        Returns:
            bool: True if fully expanded, False otherwise.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def rollout(self) -> float:
        """
        Simulate a game from the current state and return a reward.

        Returns:
            float: The reward from the simulation.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def get_legal_actions(self) -> List[Action]:
        raise NotImplementedError("This method should be implemented in a subclass.")


class HanabiNode(MCTSNode):
    """
    A specialized MCTS Node for the Hanabi game using structured actions.
    """

    def is_fully_expanded(self) -> bool:
        """
        Check if the node is fully expanded (all possible actions explored).

        Returns:
            bool: True if fully expanded, False otherwise.
        """

        return len(self.children) >= len(self.get_legal_actions())


    def get_legal_actions(self, state: GameState = None, player=None) -> List[Action]:
        """
        Get all possible legal actions in the current game state.

        Returns:
            List[Action]: A list of possible actions.
        """
        if state is None:
            state = self.state
        actions = Rules.get_rules_moves(state,state.player)
        if len(actions) == 0:
            actions = [Action(action_type="discard", player=state.player, card_idx=0)]
        return actions

    def expand(self, action: Action) -> "HanabiNode":
        """
        Expand the node by performing an action and adding a new child.

        Args:
            action: The Action object representing the move.

        Returns:
            HanabiNode: The newly created child node.
        """
        next_state = self.state.__deepcopy__({})
        if action.action_type == "play":
            next_state.play_card(action.player, action.card_idx)
        elif action.action_type == "discard":
            next_state.discard_card(action.player, action.card_idx)
        elif action.action_type == "hint":
            next_state.give_hint(action.destination, action.hint_type, action.hint_value)

        child_node = HanabiNode(state=next_state, action=action, parent=self,
                                player=(self.player + 1) % len(self.state.hands))
        self.children.append(child_node)
        return child_node

    def get_random_action(self, state: GameState = None, player: int = None) -> Action:
        """
        Select a random legal action for the current player without generating all actions.

        Args:
            state: The game state to consider. If None, use the current state.
            player: The index of the player to consider. If None, use the current player.

        Returns:
            Action: A randomly selected legal action.
        """
        return random.choice(self.get_legal_actions(state, player))
        # if state is None:
        #     state = self.state
        # if player is None:
        #     player = self.player
        #
        # # Cache state variables to reduce repeated access
        # board = state.board
        # hands = state.hands[player]
        # hints = state.hints
        # plays = []
        #
        # # Check for a playable action
        # for card_idx, card in enumerate(hands):
        #     if (card.is_fully_determined() and board[card.color] == card.rank - 1) or (card.rank_known and np.all(state.board[np.where(card.color_options == 1)] == card.rank - 1)) :
        #         return Action(action_type="play", player=player, card_idx=card_idx)
        #     elif ( self.state.errors < 3
        #         and card.rank_known
        #         and not card.color_known
        #         and np.any(self.state.board == card.rank - 1)
        #     ):
        #         plays.append( Action(action_type="play", player=player, card_idx=card_idx))
        #
        # if random.random()<0.3 and len(plays)>0:
        #     return random.choice(plays)
        # # Randomly decide between discard or hint actions
        # if random.random() < 0.5:  # 50% chance to discard
        #     card_idx = random.randint(0, len(hands) - 1)
        #     return Action(action_type="discard", player=player, card_idx=card_idx)
        #
        # # If hints are available, generate a random hint
        # if hints < MAX_HINTS:
        #     destination = random.choice([idx for idx in range(len(state.hands)) if idx != player])
        #     if random.random() < 0.5:  # 50% chance to hint by color
        #         hint_value = random.randint(0, len(board) - 1)  # Random color
        #         return Action(action_type="hint", player=player, destination=destination, hint_type="color",
        #                       hint_value=hint_value)
        #     else:  # Hint by rank
        #         hint_value = random.randint(1, 5)  # Random rank
        #         return Action(action_type="hint", player=player, destination=destination, hint_type="value",
        #                       hint_value=hint_value)
        #
        # # Default to discard if no hints are available
        # card_idx = random.randint(0, len(hands) - 1)
        # return Action(action_type="discard", player=player, card_idx=card_idx)

    def rollout(self) -> float:
        """
        Perform a simulation (rollout) from the current state.

        Returns:
            float: The reward from the simulation.
        """
        current_state = copy.deepcopy(self.state)
        while not current_state.game_ended()[0]:
            action = self.get_random_action(current_state, current_state.player)
            if action.action_type == "play":
                current_state.play_card(action.player, action.card_idx)
            elif action.action_type == "discard":
                current_state.discard_card(action.player, action.card_idx)
            elif action.action_type == "hint":
                current_state.give_hint(action.destination, action.hint_type, action.hint_value)

        _, score = current_state.game_ended()
        return score

    def select_random_action(self, legal_actions: List[Action]) -> Action:
        """
        Select a random action from the list of legal actions.

        Args:
            legal_actions: A list of Action objects.

        Returns:
            Action: A randomly chosen action.
        """
        return random.choice(legal_actions)
