import random
import sys
import copy

from hanabi import HINT_NUMBER
from players import Player, InnerStatePlayer, OuterStatePlayer, SelfRecognitionPlayer, IntentionalPlayer, \
    SamplingRecognitionPlayer, Action, ALL_COLORS, COUNTS, COLORNAMES, PLAY, DISCARD, HINT_COLOR

# Constants for game setup
GREEN, YELLOW, WHITE, BLUE, RED = 0, 1, 2, 3, 4
ALL_COLORS = [GREEN, YELLOW, WHITE, BLUE, RED]
COLORNAMES = ["green", "yellow", "white", "blue", "red"]
COUNTS = [3, 2, 2, 2, 1]


# Helper functions for game mechanics
def make_deck():
    deck = []
    for col in ALL_COLORS:
        for num, cnt in enumerate(COUNTS):
            for i in range(cnt):
                deck.append((col, num + 1))
    random.shuffle(deck)
    return deck


def initial_knowledge():
    knowledge = []
    for col in ALL_COLORS:
        knowledge.append(COUNTS[:])
    return knowledge


def playable(possible, board):
    for (col, nr) in possible:
        if board[col][1] + 1 != nr:
            return False
    return True


def discardable(possible, board):
    for (col, nr) in possible:
        if board[col][1] < nr:
            return False
    return True


class Game:
    def __init__(self, players = 2, log=sys.stdout, format=0):
        self.players = players
        self.hits = 3
        self.hints = 8
        self.current_player = 0
        self.board = [(c, 0) for c in ALL_COLORS]
        self.played = []
        self.deck = make_deck()
        self.hands = []
        self.knowledge = []
        self.trash = []
        self.log = log
        self.format = format
        self.turn = 1

        # Initialize player hands and knowledge
        for player in players:
            self.hands.append([])
            self.knowledge.append(initial_knowledge())
            self.draw_card(len(self.hands) - 1)  # Draw a card for each player

        if self.format:
            print(self.deck, file=self.log)

    def draw_card(self, pnr):
        if not self.deck:
            return
        self.hands[pnr].append(self.deck.pop(0))  # Draw the top card from the deck

    def perform(self, action):
        # Inform all players about the action
        for p in self.players:
            p.inform(action, self.current_player, self)

        if self.format:
            print(f"MOVE: {self.current_player} {action}", file=self.log)

        if action.type == HINT_COLOR:
            self.hints -= 1
            print(
                f"{self.players[self.current_player].name} hints {self.players[action.pnr].name} about all their {COLORNAMES[action.col]} cards",
                file=self.log)
            self.update_knowledge_hints(action)
        elif action.type == HINT_NUMBER:
            self.hints -= 1
            print(
                f"{self.players[self.current_player].name} hints {self.players[action.pnr].name} about all their {action.num} cards",
                file=self.log)
            self.update_number_hints(action)
        elif action.type == PLAY:
            self.execute_play(action)
        elif action.type == DISCARD:
            self.execute_discard(action)

    def update_knowledge_hints(self, action):
        # Update knowledge based on color hint
        for (col, num), knowledge in zip(self.hands[action.pnr], self.knowledge[action.pnr]):
            if col == action.col:
                for i in range(len(knowledge)):
                    if i != col:
                        knowledge[i] = 0
            else:
                for i in range(len(knowledge[action.col])):
                    knowledge[action.col][i] = 0

    def update_number_hints(self, action):
        # Update knowledge based on number hint
        for (col, num), knowledge in zip(self.hands[action.pnr], self.knowledge[action.pnr]):
            if num == action.num:
                for k in knowledge:
                    for i in range(len(COUNTS)):
                        if i + 1 != num:
                            k[i] = 0
            else:
                for k in knowledge:
                    k[action.num - 1] = 0

    def execute_play(self, action):
        (col, num) = self.hands[self.current_player][action.cnr]
        print(f"{self.players[self.current_player].name} plays {COLORNAMES[col]} {num}", file=self.log)
        if self.board[col][1] == num - 1:
            self.board[col] = (col, num)
            self.played.append((col, num))
            if num == 5:
                self.hints += 1  # Giving back a hint for playing 5
                self.hints = min(self.hints, 8)  # Ensure it doesn't exceed 8
            print(f"successfully! Board is now {self.board}", file=self.log)
        else:
            self.trash.append((col, num))
            self.hits -= 1  # Decrease hits because of a failure
            print(f"and fails. Board was {self.board}", file=self.log)

        # Remove the card from the player's hand
        del self.hands[self.current_player][action.cnr]
        del self.knowledge[self.current_player][action.cnr]
        self.draw_card(self.current_player)
        print(f"{self.players[self.current_player].name} now has {self.hands[self.current_player]}", file=self.log)

    def execute_discard(self, action):
        self.hints += 1
        self.hints = min(self.hints, 8)  # Ensure it doesn't exceed 8
        discarded_card = self.hands[self.current_player][action.cnr]
        self.trash.append(discarded_card)
        print(f"{self.players[self.current_player].name} discards {COLORNAMES[discarded_card[0]]} {discarded_card[1]}", file=self.log)
        print(f"trash is now {self.trash}", file=self.log)

        # Remove the card from the player's hand
        del self.hands[self.current_player][action.cnr]
        del self.knowledge[self.current_player][action.cnr]
        self.draw_card(self.current_player)
        print(f"{self.players[self.current_player].name} now has {self.hands[self.current_player]}", file=self.log)

    def valid_actions(self):
        valid = []
        # Play and discard actions
        for i in range(len(self.hands[self.current_player])):
            valid.append(Action(PLAY, cnr=i))
            valid.append(Action(DISCARD, cnr=i))

        # Hint actions
        if self.hints > 0:
            for i, p in enumerate(self.players):
                if i != self.current_player:
                    for col in set([col_num[0] for col_num in self.hands[i]]):
                        valid.append(Action(HINT_COLOR, pnr=i, col=col))
                    for num in set([col_num[1] for col_num in self.hands[i]]):
                        valid.append(Action(HINT_NUMBER, pnr=i, num=num))
        return valid

    def run(self, turns=-1):
        self.turn = 1
        while not self.done() and (turns < 0 or self.turn < turns):
            self.turn += 1
            if not self.deck:  # Check for extra turns when deck is empty
                print("No more cards in the deck.", file=self.log)

            action = self.players[self.current_player].get_action(
                self.current_player,
                [h for i, h in enumerate(self.hands) if i != self.current_player],
                self.knowledge,
                self.trash,
                self.played,
                self.board,
                self.valid_actions(),
                self.hints
            )
            self.perform(action)
            self.current_player = (self.current_player + 1) % len(self.players)  # Move to the next player

        print("Game done! Hits left:", self.hits, file=self.log)
        points = self.score()
        print("Points:", points, file=self.log)
        return points

    def score(self):
        return sum([self.board[col][1] for col in ALL_COLORS if self.board[col][1] < 5])

    def done(self):
        if self.hits == 0 or all(num == 5 for _, num in self.board):
            return True
        return False

# Main game execution block
if __name__ == "__main__":
    # Set up players and initiate the game
    players = [
        InnerStatePlayer("Alice", 0),
        OuterStatePlayer("Bob", 1),
        SelfRecognitionPlayer("Charlie", 2),
        IntentionalPlayer("Dana", 3)
    ]
    game = Game(players)
    game.run()