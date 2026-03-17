"""
Pure Python Hanabi Environment with Text-based State Representation.

This implements a simplified Hanabi environment that produces text descriptions
of the game state, matching the format used in the Sudhakar et al. paper.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class Color(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    WHITE = 3
    BLUE = 4


COLOR_NAMES = ["red", "yellow", "green", "white", "blue"]
RANK_NAMES = ["1", "2", "3", "4", "5"]

CARD_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


@dataclass
class Card:
    color: int
    rank: int
    
    def __repr__(self):
        return f"{COLOR_NAMES[self.color]} {self.rank + 1}"


@dataclass 
class CardKnowledge:
    """What a player knows about a card in their hand."""
    possible_colors: List[bool] = field(default_factory=lambda: [True] * 5)
    possible_ranks: List[bool] = field(default_factory=lambda: [True] * 5)
    
    def apply_color_hint(self, color: int, matches: bool):
        if matches:
            self.possible_colors = [i == color for i in range(5)]
        else:
            self.possible_colors[color] = False
    
    def apply_rank_hint(self, rank: int, matches: bool):
        if matches:
            self.possible_ranks = [i == rank for i in range(5)]
        else:
            self.possible_ranks[rank] = False


@dataclass
class HanabiState:
    """Complete state of a Hanabi game."""
    num_players: int
    hand_size: int
    hands: List[List[Card]]
    knowledge: List[List[CardKnowledge]]
    deck: List[Card]
    fireworks: List[int]  # Current value for each color (0-5)
    hint_tokens: int
    life_tokens: int
    discard_pile: List[Card]
    current_player: int
    last_action: Optional[str]
    turns_remaining: Optional[int]  # None until deck empties, then counts down
    score: int
    terminal: bool


class HanabiEnv:
    """Pure Python Hanabi environment."""
    
    def __init__(
        self,
        num_players: int = 2,
        hand_size: int = 5,
        num_colors: int = 5,
        num_ranks: int = 5,
        max_hint_tokens: int = 8,
        max_life_tokens: int = 3,
        seed: Optional[int] = None,
    ):
        self.num_players = num_players
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.max_hint_tokens = max_hint_tokens
        self.max_life_tokens = max_life_tokens
        self.rng = random.Random(seed)
        
        self.state: Optional[HanabiState] = None
        self._action_space = self._build_action_space()
        
    def _build_action_space(self) -> List[Dict]:
        """Build the action space for Hanabi."""
        actions = []
        
        # Discard actions: D0-D4
        for i in range(self.hand_size):
            actions.append({"type": "discard", "card_idx": i})
        
        # Play actions: P0-P4
        for i in range(self.hand_size):
            actions.append({"type": "play", "card_idx": i})
        
        # Hint color actions
        for color in range(self.num_colors):
            actions.append({"type": "hint_color", "color": color})
        
        # Hint rank actions
        for rank in range(self.num_ranks):
            actions.append({"type": "hint_rank", "rank": rank})
        
        return actions
    
    def num_actions(self) -> int:
        return len(self._action_space)
    
    def _create_deck(self) -> List[Card]:
        """Create and shuffle the deck."""
        deck = []
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                count = CARD_COUNTS[rank + 1]
                for _ in range(count):
                    deck.append(Card(color, rank))
        self.rng.shuffle(deck)
        return deck
    
    def reset(self) -> HanabiState:
        """Reset the environment to initial state."""
        deck = self._create_deck()
        
        hands = []
        knowledge = []
        for _ in range(self.num_players):
            hand = [deck.pop() for _ in range(self.hand_size)]
            hands.append(hand)
            knowledge.append([CardKnowledge() for _ in range(self.hand_size)])
        
        self.state = HanabiState(
            num_players=self.num_players,
            hand_size=self.hand_size,
            hands=hands,
            knowledge=knowledge,
            deck=deck,
            fireworks=[0] * self.num_colors,
            hint_tokens=self.max_hint_tokens,
            life_tokens=self.max_life_tokens,
            discard_pile=[],
            current_player=0,
            last_action=None,
            turns_remaining=None,
            score=0,
            terminal=False,
        )
        return self.state
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices."""
        if self.state.terminal:
            return []
        
        legal = []
        
        # Discard/Play always legal if you have cards
        current_hand_size = len(self.state.hands[self.state.current_player])
        for i in range(current_hand_size):
            legal.append(i)  # Discard
            legal.append(i + self.hand_size)  # Play
        
        # Hints legal if hint tokens > 0
        if self.state.hint_tokens > 0:
            base_idx = 2 * self.hand_size
            # Check which hints are valid (must point to at least one card)
            other_player = (self.state.current_player + 1) % self.num_players
            other_hand = self.state.hands[other_player]
            
            for color in range(self.num_colors):
                if any(card.color == color for card in other_hand):
                    legal.append(base_idx + color)
            
            for rank in range(self.num_ranks):
                if any(card.rank == rank for card in other_hand):
                    legal.append(base_idx + self.num_colors + rank)
        
        return legal
    
    def step(self, action_idx: int) -> Tuple[HanabiState, float, bool]:
        """Take an action in the environment."""
        if self.state.terminal:
            return self.state, 0.0, True
        
        action = self._action_space[action_idx]
        reward = 0.0
        current_player = self.state.current_player
        
        if action["type"] == "discard":
            card_idx = action["card_idx"]
            card = self.state.hands[current_player].pop(card_idx)
            self.state.knowledge[current_player].pop(card_idx)
            self.state.discard_pile.append(card)
            self.state.hint_tokens = min(self.state.hint_tokens + 1, self.max_hint_tokens)
            self.state.last_action = f"discard {card}"
            self._draw_card(current_player)
            
        elif action["type"] == "play":
            card_idx = action["card_idx"]
            card = self.state.hands[current_player].pop(card_idx)
            self.state.knowledge[current_player].pop(card_idx)
            
            if self.state.fireworks[card.color] == card.rank:
                self.state.fireworks[card.color] += 1
                self.state.score += 1
                reward = 1.0
                if card.rank == 4:  # Completed a stack
                    self.state.hint_tokens = min(self.state.hint_tokens + 1, self.max_hint_tokens)
                self.state.last_action = f"play {card} success"
            else:
                self.state.life_tokens -= 1
                self.state.discard_pile.append(card)
                self.state.last_action = f"play {card} fail"
                if self.state.life_tokens == 0:
                    self.state.terminal = True
            
            self._draw_card(current_player)
            
        elif action["type"] == "hint_color":
            color = action["color"]
            other_player = (current_player + 1) % self.num_players
            self.state.hint_tokens -= 1
            
            for i, card in enumerate(self.state.hands[other_player]):
                self.state.knowledge[other_player][i].apply_color_hint(
                    color, card.color == color
                )
            self.state.last_action = f"hint {COLOR_NAMES[color]} to player {other_player}"
            
        elif action["type"] == "hint_rank":
            rank = action["rank"]
            other_player = (current_player + 1) % self.num_players
            self.state.hint_tokens -= 1
            
            for i, card in enumerate(self.state.hands[other_player]):
                self.state.knowledge[other_player][i].apply_rank_hint(
                    rank, card.rank == rank
                )
            self.state.last_action = f"hint {rank + 1} to player {other_player}"
        
        # Check for game end conditions
        if self.state.score == self.num_colors * self.num_ranks:
            self.state.terminal = True
        
        if self.state.turns_remaining is not None:
            self.state.turns_remaining -= 1
            if self.state.turns_remaining == 0:
                self.state.terminal = True
        
        # Move to next player
        self.state.current_player = (current_player + 1) % self.num_players
        
        return self.state, reward, self.state.terminal
    
    def _draw_card(self, player: int):
        """Draw a card from deck if available."""
        if len(self.state.deck) > 0:
            card = self.state.deck.pop()
            self.state.hands[player].append(card)
            self.state.knowledge[player].append(CardKnowledge())
        elif self.state.turns_remaining is None:
            self.state.turns_remaining = self.num_players


class TextHanabiEnv(HanabiEnv):
    """Hanabi environment with text-based state representation."""
    
    def get_text_state(self, observer: int) -> Dict[str, str]:
        """
        Get text representation of state from observer's perspective.
        
        Returns a dictionary with the 7 sections from the milestone:
        - life_tokens
        - hint_tokens  
        - fireworks
        - own_hand (what observer knows about their hand)
        - opponent_hand (full info about other players' hands)
        - discards
        - last_action
        """
        sections = {}
        
        # Life tokens
        sections["life_tokens"] = f"Lives: {self.state.life_tokens}"
        
        # Hint tokens
        sections["hint_tokens"] = f"Hints: {self.state.hint_tokens}"
        
        # Fireworks
        fw_strs = []
        for color in range(self.num_colors):
            fw_strs.append(f"{COLOR_NAMES[color]}: {self.state.fireworks[color]}")
        sections["fireworks"] = "Fireworks: " + ", ".join(fw_strs)
        
        # Own hand (from observer's knowledge)
        own_hand_strs = []
        for i, knowledge in enumerate(self.state.knowledge[observer]):
            colors = [COLOR_NAMES[c] for c in range(5) if knowledge.possible_colors[c]]
            ranks = [str(r + 1) for r in range(5) if knowledge.possible_ranks[r]]
            own_hand_strs.append(f"Card {i}: colors=[{','.join(colors)}] ranks=[{','.join(ranks)}]")
        sections["own_hand"] = "My hand: " + "; ".join(own_hand_strs)
        
        # Opponent's hand (full information)
        opp_hand_strs = []
        for p in range(self.num_players):
            if p != observer:
                for i, card in enumerate(self.state.hands[p]):
                    opp_hand_strs.append(f"P{p} Card {i}: {card}")
        sections["opponent_hand"] = "Partner's hand: " + "; ".join(opp_hand_strs)
        
        # Discards
        if self.state.discard_pile:
            discard_strs = [str(card) for card in self.state.discard_pile]
            sections["discards"] = "Discards: " + ", ".join(discard_strs)
        else:
            sections["discards"] = "Discards: none"
        
        # Last action
        sections["last_action"] = f"Last action: {self.state.last_action or 'none'}"
        
        return sections
    
    def get_full_text_state(self, observer: int) -> str:
        """Get concatenated text state."""
        sections = self.get_text_state(observer)
        return " | ".join([
            sections["life_tokens"],
            sections["hint_tokens"],
            sections["fireworks"],
            sections["own_hand"],
            sections["opponent_hand"],
            sections["discards"],
            sections["last_action"],
        ])
    
    def get_action_text(self, action_idx: int) -> str:
        """Get text description of an action."""
        action = self._action_space[action_idx]
        
        if action["type"] == "discard":
            return f"discard card {action['card_idx']}"
        elif action["type"] == "play":
            return f"play card {action['card_idx']}"
        elif action["type"] == "hint_color":
            return f"hint {COLOR_NAMES[action['color']]}"
        elif action["type"] == "hint_rank":
            return f"hint {action['rank'] + 1}"
        return "unknown"
