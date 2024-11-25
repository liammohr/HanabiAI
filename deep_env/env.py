from typing import Any, Dict, List, Tuple
import gym
import numpy as np
import scipy.linalg
from gym.spaces import Discrete, MultiDiscrete

from game_state import GameState, MAX_HINTS, Action

from utils import Deck, Trash, Color


class HanabiEnv(gym.Env):
    def __init__(self, num_players: int = 2):
        super().__init__()
        self.num_players = num_players
        self.game_state = GameState(root=0, hands=None, hints=MAX_HINTS, errors=0, deck=Deck(), trash=Trash())
        self.action_space = MultiDiscrete(
            [3, self.num_players, 5, self.num_players, 2, 5])  # Encodes all possible actions

        # Calculate observation size based on _get_observation
        self.observation_size = (
                (num_players * 5 * 26) +  # Hand probabilities (25 probabilities + 1 for None) for 2 players
                len(Color) +  # Board progress (one per color)
                2 +  # Hints and errors
                (num_players * 25) +  # Trash and remaining cards per type
                25  # Deck count (remaining cards in play)
        )

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_size,),
            dtype=np.float32
        )

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment to the initial state.

        Returns:
            Observation: A dictionary representing the initial game state.
        """
        self.game_state = GameState(root=0, hands=None, hints=MAX_HINTS, errors=0, deck=Deck(), trash=Trash())
        return self._get_observation()

    def get_legal_actions(self) -> np.ndarray:
        """
        Generate a mask for all possible actions, where 1 indicates the action is legal,
        and 0 indicates it is illegal.

        Returns:
            np.ndarray: A binary mask array for all possible actions.
        """
        num_actions = 20  # Total number of possible actions
        action_mask = np.zeros(num_actions, dtype=np.int32)

        # Check for discard and play legality (cards in hand)
        for card_idx in range(5):  # Max 5 cards in hand
            if self.game_state.hands[self.game_state.player][card_idx] is not None:
                action_mask[card_idx] = 1  # Legal discard
                action_mask[card_idx + 5] = 1  # Legal play
        action_mask[10:] = int(MAX_HINTS > self.game_state.hints)
        return action_mask

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Executes an action in the environment.

        Args:
            action: A list representing the action [action_type, player, card_idx, destination, hint_type, hint_value].

        Returns:
            observation: The new observation after the action.
            reward: The reward for the action.
            done: Whether the game has ended.
            info: Additional information about the step.
        """
        action_type, player, card_idx, destination, hint_type, hint_value = self.decode_action(action)

        previous_board = self.game_state.board.copy()
        # Reward Function
        reward = 0

        # Reward for progress


        # Penalty for errors
        if self.game_state.errors > 0:
            reward -= self.game_state.errors * 2

        # Penalty for discarding playable cards
        if action_type == 1:  # Discard action
            discarded_card = self.game_state.hands[player][card_idx]
            if discarded_card.rank == self.game_state.board[discarded_card.color] + 1:
                reward -= 3  # Penalty for discarding a card that could progress the board

        # Bonus for end-game success

        if action_type == 0:  # Play card

            self.game_state.play_card(player, card_idx)
        elif action_type == 1:  # Discard card
            self.game_state.discard_card(player, card_idx)
        elif action_type == 2:  # Give hint
            self.game_state.give_hint(destination, hint_type, hint_value)

        for i, progress in enumerate(self.game_state.board):
            if progress > previous_board[i]:
                reward += (progress - previous_board[i])  # Reward for incremental progress
                if progress == 5:  # Bonus for completing a color
                    reward += 5
        # Check game state
        done, score = self.game_state.game_ended()
        if done:
            reward += score  # Reward based on the final score


        return self._get_observation(), reward, done, {}

    def render(self, mode: str = "human") -> None:
        """
        Renders the current state of the environment.
        """
        print("Board:", self.game_state.board)
        print("Hands:", [[str(card) for card in hand] for hand in self.game_state.hands])
        print("Hints:", self.game_state.hints)
        print("Errors:", self.game_state.errors)
        print("Trash:", [str(card) for card in self.game_state.trash])

    def decode_action(self, action_index: int) -> Tuple:
        """
        Decodes a flattened action index into an Action object based on specified rules:
        - If action_index < 5: Discard action.
        - If 5 <= action_index < 10: Play action.
        - If action_index >= 10: Hint action (decoded further).

        Args:
            action_index (int): The flattened action index.
            current_player (int): The index of the current player performing the action.
            num_players (int): The number of players in the game.

        Returns:
            Action: Decoded action object.
        """
        if action_index < 5:
            action_type = 0
            card_idx = action_index  # Card to discard
            return action_type, self.game_state.player, card_idx, None, None, None

        elif 5 <= action_index < 10:
            action_type = 1
            card_idx = action_index - 5  # Card to play
            return action_type, self.game_state.player, card_idx, None, None, None

        else:  # Hint action
            action_type = 2
            hint_range_start = 10
            hint_idx = action_index - hint_range_start

            # Decoding hint into parameters
            destination = 1 - self.game_state.player  # Target player for the hint
            hint_type = "color" if hint_idx % 2 == 0 else "value"  # Alternate between color and value
            hint_value = hint_idx % 5  # Value or color index of the hint

            return action_type, self.game_state.player, 0, destination, hint_type, hint_value

    def _get_observation(self) -> np.ndarray:
        """
        Encodes the current state as an observation, incorporating probabilities derived
        from the remaining cards in play.

        Returns:
            A flattened numpy array representing the current observation.
        """
        num_possible_cards = 25  # 5 colors × 5 ranks
        num_cards_per_hand = 5
        remaining = self.game_state.trash.get_table()  # Remaining cards in play
        belief = np.zeros((num_cards_per_hand * 2, num_possible_cards + 1), dtype=np.float32)
        # Probabilities for the current player's hand
        # Opponent hands (fully visible to the current player)
        hand = self.game_state.hands[self.game_state.player]
        for i, card in enumerate(hand):
            if card is not None:
                one_hot = card.color_options.reshape(-1, 1).dot(card.number_options.reshape(1, -1))
                assert one_hot.shape == (5, 5)
                n = ((one_hot * remaining).flatten())
                belief[i][:25] =0 if np.linalg.norm(n) ==0 else n / np.linalg.norm(n)
            else:
                belief[i][-1] = 1

        hand = self.game_state.hands[1 - self.game_state.player]
        for i, card in enumerate(hand):
            if card is not None:
                one_hot = card.color_options.reshape(-1, 1).dot(card.number_options.reshape(1, -1))
                assert one_hot.shape == (5, 5)
                n = ((one_hot * remaining).flatten())
                belief[i][:25] =0 if np.linalg.norm(n) ==0 else n / np.linalg.norm(n)
            else:
                belief[i + 5][-1] = 1

        opponent_hands = [
            card.color_options.dot(card.color_options).flatten()

            for card in self.game_state.hands[1 - self.game_state.player]
        ]

        # Trash pile encoded as a count of remaining cards for each type

        # Combine all components into a flattened observation
        observation = np.concatenate([
            belief.flatten(),  # Current player's hand probabilities
            np.array(opponent_hands).flatten(),  # Opponent hands
            self.game_state.board.flatten(),  # Board progress
            np.array([self.game_state.hints, self.game_state.errors]),  # Hints and errors
            self.game_state.deck.get_table().flatten(),  # Trash pile counts
            self.game_state.trash.get_table().flatten()  # Remaining cards in play

        ]).astype(np.float32)

        return observation
