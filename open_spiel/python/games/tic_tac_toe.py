# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g. MCTS.
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="python_tic_tac_toe",
    long_name="Python Tic-Tac-Toe",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS)


class TicTacToeGame(pyspiel.Game):
  """A Python version of the Tic-Tac-Toe game."""

  def __init__(self, params=None, game_type="A"):
    if game_type == "A":
      game_info = _GAME_INFO

    elif game_type == "B":
      game_info = pyspiel.GameInfo(
        num_distinct_actions=_NUM_CELLS*2,
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=_NUM_CELLS)

    elif game_type == "C":
      game_info = pyspiel.GameInfo(
        num_distinct_actions=3**_NUM_CELLS,
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=_NUM_CELLS)

    super().__init__(_GAME_TYPE, game_info, params or dict())
    self.game_type = game_type


  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    if(self.game_type == "A"):
      return TicTacToeState(self)
    elif(self.game_type == "B"):
      return TicTacToeBState(self)
    elif(self.game_type == "C"):
      return TicTacToeCState(self)
    else:
      raise Exception("Unknown game type")

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class TicTacToeState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    # return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]
    return [a for a in range(_NUM_CELLS)]

  def legal_actions_real(self, player):
    return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    illegal = False
    if self.board[_coord(action)] != ".":
      illegal = True
      
    self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"
    if illegal:
      self._is_terminal = True
      self._player0_score = -1.0 if self._cur_player == 0 else 1.0
    elif _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    row, col = _coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class TicTacToeBState(pyspiel.State):
  """Overwrite + Any symbol"""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    # return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]
    return [a for a in range(_NUM_CELLS * 2)]

  def legal_actions_real(self, player):
    shape_offset = 0 if player == 0 else _NUM_CELLS
    return [shape_offset + a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _get_pos(self, action):
    return action % _NUM_CELLS
  
  def _get_shape(self, action):
    return "x" if (action // _NUM_CELLS == 0) else "o"

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    pos = self._get_pos(action)
    shape = self._get_shape(action)

    illegal = False
    correct_shape = "x" if self._cur_player == 0 else "o"
    if self.board[_coord(pos)] != "." or shape != correct_shape:
      illegal = True
    
    self.board[_coord(pos)] = shape
    # self.board[_coord(pos)] = "x" if self._cur_player == 0 else "o"
    if illegal:
      self._is_terminal = True
      self._player0_score = -1.0 if self._cur_player == 0 else 1.0
    elif _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    pos = self._get_pos(action)
    shape = self._get_shape(action)
    row, col = _coord(pos)
    return "{}({},{})".format(shape, row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


import string
digs = string.digits + string.ascii_letters

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)

def int2base_fixed(x, base, width):
  return int2base(x, base).zfill(width)

def convert_board_state_str(state):
    result = []
    for idx in range(9):
        pos = _coord(idx)
        if state[pos] == 'x':
            result.append(str(1))
        elif state[pos] == '.':
            result.append(str(0))
        elif state[pos] == 'o':
            result.append(str(-1))
        else:
            raise Exception("invalid cell state")
    return ",".join(result)


def is_valid_move(current_state, next_state):
    current_parts = str(current_state).split(',')
    next_parts = str(next_state).split(',')
    # print(current_parts)
    # print(next_parts)
    assert len(current_parts) == len(next_parts)

    changed_count = 0
    new_count = 0

    num_xs = len([c for c in current_parts if c == '1'])
    num_os = len([c for c in current_parts if c == '-1'])
    correct_move = None
    if(num_xs == num_os):
        correct_move = '1'
    else:
        correct_move = '-1'

    shape = None
    for idx in range(len(current_parts)):
        if current_parts[idx] == '0' and next_parts[idx] != '0':
            new_count += 1
            shape = next_parts[idx]
        elif current_parts[idx] != '0' and next_parts[idx] != current_parts[idx]:
            # print('idx', idx, 'curr', current_parts[idx], 'next', next_parts[idx])
            changed_count += 1
    return changed_count == 0 and new_count == 1 and shape == correct_move

class TicTacToeCState(pyspiel.State):
  """Change board state to whatever"""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    # return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]
    return [a for a in range(3**(_NUM_CELLS))]

  def _get_board_from_action(self, action):
    board_str = int2base_fixed(action, 3, _NUM_CELLS)
    assert(len(board_str) == _NUM_CELLS)
    res = np.full((_NUM_ROWS, _NUM_COLS), ".")
    cell_vals = ["o", ".", "x"]
    for i in range(len(board_str)):
      res[_coord(i)] = cell_vals[int(board_str[i])]
    
    return res

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    new_board = self._get_board_from_action(action)

    legal = is_valid_move(convert_board_state_str(self.board), convert_board_state_str(new_board))
    
    self.board = new_board
    # self.board[_coord(pos)] = "x" if self._cur_player == 0 else "o"
    if not legal:
      self._is_terminal = True
      self._player0_score = -1.0 if self._cur_player == 0 else 1.0
    elif _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    # pos = self._get_pos(action)
    # shape = self._get_shape(action)
    # row, col = _coord(pos)
    return "{}".format(action)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _line_value(line):
  """Checks a possible line, returning the winning symbol if any."""
  if all(line == "x") or all(line == "o"):
    return line[0]


def _line_exists(board):
  """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
  return (_line_value(board[0]) or _line_value(board[1]) or
          _line_value(board[2]) or _line_value(board[:, 0]) or
          _line_value(board[:, 1]) or _line_value(board[:, 2]) or
          _line_value(board.diagonal()) or
          _line_value(np.fliplr(board).diagonal()))


def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TicTacToeGame)
