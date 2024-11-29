import copy
from typing import List, Tuple, Optional

import numpy as np

from utils import Card, Deck, Trash, Color

MAX_HINTS = 8
SCORE_3_ERRORS = 0.1
MAX_ERRORS = 3
DEFAULT_PLAYERS = 2


class GameState:
    """
    Represents a determined game state
    """

    def __init__(self, root: int, hands: List[List[Card]] = None, hints: int = 0, errors: int = 0, deck: Deck = Deck(),
                 trash: Trash = Trash(),
                 board: np.array = None):
        if hands is None:
            hands = []
            for i in range(DEFAULT_PLAYERS):
                hands.append([deck.draw() for _ in range(5)])

        if board is None:
            board = np.zeros(len(list(Color)))
        self.last_turn_played = np.full(len(hands), False)
        self.root: int = root
        self.player = root
        self.hands: List[List[Card]] = hands
        self.hints: int = hints
        self.errors: int = errors
        self.deck = copy.deepcopy(deck)
        self.trash = copy.deepcopy(trash)
        self.board = copy.deepcopy(board)

    def play_card(self, player: int, card_idx: int) -> None:
        """
        Track a card played by "player". The played card will be removed from the player's hand
        and added to either the board or the trash depending on its value
        (the number of tokens will be adjusted accordingly)
        A new card drawn from the deck will be appended in the last position of the player's hand

        Args:
            player: the name of the player
            card_idx: the index of the card in the player's hand
        """
        card = self.hands[player].pop(card_idx)
        if len(self.deck) > 0:
            self.deck.assert_no_reserved_cards()
            self.hands[player].append(self.deck.draw())
            assert self.hands[player][-1] is not None
        else:
            self.last_turn_played[player] = True
        if card is not None and self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            if card.rank == 5 and self.hints > 0:
                self.hints -= 1
        elif card is not None:
            self.trash.append(card)
            self.errors += 1
        self.player = (self.player + 1) % len(self.hands)

    def discard_card(self, player: int, card_idx: int) -> None:
        """
        Track a card discarded by "player". The discarded card will be removed from the player's hand
        and added to the trash (the number of tokens will be adjusted accordingly)
        A new card drawn from the deck will be appended in the last position of the player's hand

        Args:
            player: the name of the player
            card_idx: the index of the card in the player's hand
        """
        # if self.hints == 0:
        #     raise RuntimeError("No used hint tokens")
        card = self.hands[player].pop(card_idx)
        self.trash.append(card)
        if len(self.deck) > 0:
            self.deck.assert_no_reserved_cards()
            self.hands[player].append(self.deck.draw())
        else:
            self.last_turn_played[player] = True
        self.hints = max(self.hints - 1, 0)
        self.player = (self.player + 1) % len(self.hands)

    def give_hint(self, destination: int, hint_type: str, hint_value: int) -> None:
        """
        This works asssuming that all the cards in all the players' hands have a defined rank and color
        (either known or not)
        """
        # if self.hints == MAX_HINTS:
        #     raise RuntimeError("Maximum number of hints already reached")
        hand = self.hands[destination]
        for card in hand:
            if card is not None:
                if hint_type == "value":
                    card.reveal_rank(hint_value)
                elif hint_type == "color":
                    card.reveal_color(hint_value)
        if len(self.deck) <= 0:
            self.last_turn_played[:] = True
        self.hints = min(self.hints + 1, MAX_HINTS)
        self.player = (self.player + 1) % len(self.hands)

    def redeterminize_hand(self, player: int, bypass=False) -> None:
        """
        Redeterminizes player's hand before exiting the node associated with their move.

        Args:
            player: the name of the player
        """
        hand = self.hands[player]
        new_card = None
        if player == self.root and not bypass:
            return
        self.deck.add_cards(hand, ignore_fd=True)
        success = False
        while not success:
            success = True
            self.deck.reserve_semi_determined_cards(hand)

            new_hand = []
            for idx, card in enumerate(hand):
                if card is not None and card.is_fully_determined():
                    new_hand.append(card)
                else:
                    if card is not None:
                        rank = card.rank if card.rank_known else None
                        color = card.color if card.color_known else None
                        new_card = self.deck.draw(rank=rank, color=color, color_options=card.color_options, number_options=card.number_options)
                        if new_card is None:
                            self.deck.reset_reservations()
                            self.deck.add_cards(new_hand, ignore_fd=True)
                            success = False
                            break
                        assert new_card.rank_known == card.rank_known
                        assert new_card.color_known == card.color_known

                    # new_card.number_options = card.number_options
                    # new_card.color_options = card.color_options
                    if new_card:
                        new_hand.append(new_card)

            self.deck.assert_no_reserved_cards()
            self.hands[player] = new_hand

    def _remove_illegal_cards(self, cards: List[Card]) -> None:
        """
        Remove the illegal cards from the list (considering all the cards in the trash,
        in the player's hands and on the table)

        Args:
            cards: the list of cards to modify
        """

        for idx in range(len(cards)):
            card = cards[idx]
            if card is not None:
                if self.deck[card.rank, card.color] > 0:
                    self.deck.remove_cards([card])
                else:
                    cards[idx] = self.deck.draw()
                    assert cards[idx] is not None

    def restore_hand(self, player: int, saved_hand: List[Card], root: int) -> None:
        """
        Restore the specified hand for the specified player, removing all the "illegal" cards
        and re-determinizing their slots

        Args:
            player: the name of the player
            saved_hand: the hand to restore
        """
        if player == root:
            return

        self.deck.add_cards(self.hands[player])  # put cards back in deck
        self._remove_illegal_cards(saved_hand)  # remove inconsistencies
        if len(self.hands[player]) > len(saved_hand):
            self.deck.assert_no_reserved_cards()
            saved_hand.append(self.deck.draw())
            assert saved_hand[-1] is not None
            # assert len(self.hands[player]) == len(saved_hand)  # at most 1 card
        self.hands[player] = saved_hand

    def game_ended(self) -> Tuple[bool, Optional[float]]:
        """
        Checks if the game is ended for some reason. If it's ended, it returns True and the score of the game.
        If the game isn't ended, it returns False, None
        """
        if self.errors >= MAX_ERRORS:
            return True, sum(self.board) * SCORE_3_ERRORS
        # if self.board == self.trash.maxima:
        if np.all(self.board == 5):
            return True, sum(self.board)
        if all(self.last_turn_played):
            return True, sum(self.board)
        return False, None

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.root = copy.copy(self.root)
        result.hands = copy.deepcopy(self.hands)
        result.board = np.copy(self.board)
        result.trash = copy.deepcopy(self.trash)
        result.deck = copy.deepcopy(self.deck)
        result.hints = self.hints
        result.errors = self.errors
        result.last_turn_played = copy.deepcopy(self.last_turn_played)
        result.player = self.player
        return result

    def available_hints(self):
        return MAX_HINTS - self.hints

    def get_next_player(self, destination):
        return (destination+1)%len(self.hands)


class Action:
    """
    Represents an action in the Hanabi game.
    """

    def __init__(self, action_type: str, player: int, card_idx: Optional[int] = None,
                 destination: Optional[int] = None, hint_type: Optional[str] = None,
                 hint_value: Optional[int] = None):
        """
        Initialize an action.

        Args:
            action_type: The type of action ('play', 'discard', or 'hint').
            player: The index of the player performing the action.
            card_idx: The index of the card in the player's hand (if applicable).
            destination: The index of the player receiving the hint (if applicable).
            hint_type: The type of hint ('color' or 'value', if applicable).
            hint_value: The value or color of the hint (if applicable).
        """
        self.action_type = action_type
        self.player = player
        self.card_idx = card_idx
        self.destination = destination
        self.hint_type = hint_type
        self.hint_value = hint_value

    def __repr__(self):
        """
        Return a string representation of the action.
        """
        return (f"Action(type={self.action_type}, player={self.player}, card_idx={self.card_idx}, "
                f"destination={self.destination}, hint_type={self.hint_type}, hint_value={self.hint_value})")

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type and
                self.player == other.player and
                self.destination == other.destination and
                self.hint_type == other.hint_type and
                self.hint_value == other.hint_value and
                self.card_idx == other.card_idx)

    def __hash__(self):
        return hash((self.action_type, self.player, self.destination,
                     self.hint_type, self.hint_value, self.card_idx))