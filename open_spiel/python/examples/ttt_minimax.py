import pyspiel

import dill
from treelib import Node, Tree
import numpy as np
import pandas as pd


from tqdm import tqdm
import itertools


class Memoize_tree:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}                                      # Create our empty memo buffer

    def __call__(self, *args):
        function_call_hash = args[1:]                       # Note we skip the first argument, this is the tree that is always the same. Adding this would slow down the hashing procedure
        if function_call_hash not in self.memo:             # Check if the function has been called before
            self.memo[function_call_hash] = self.fn(*args)  # Store the result of the function call
        return self.memo[function_call_hash]                # return the result from the memo dictionary

@Memoize_tree   # Decorate the minimax algorithm
def minmax_tt(tree, current_id, is_max):
    current_node = tree[current_id] 
    if current_node.data.is_endstate():
        return current_node.data.get_value()
    children_of_current_id = tree.children(current_id)
    scores = [minmax_tt(tree, child.identifier, not is_max) for child in children_of_current_id]
    if is_max:
        return max(scores)
    else:
        return min(scores)




class MinimaxBot(pyspiel.Bot):
  """Minimax bot for tic tac toe
  """

  def __init__(self, game, game_type, tree_file):
    
    pyspiel.Bot.__init__(self)

    self._game = game
    self.game_type = game_type
    params = game.get_parameters()
    # if "board_size" in params:
    #  Use board size
    #     print('todo')

    self.cur_state = ''
    self.letters_to_move = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    with open(tree_file, 'rb') as f:
        self.tree = dill.load(f)

    # self.warmup_cache()

#   def __del__(self):
#     self.close()

  def encode_action(self, move):
      if self.game_type == "A":
        return self.letters_to_move.index(move)
      if self.game_type == "B":
        print(self.cur_state)
        pos = self.letters_to_move.index(move)
        if len(self.cur_state) % 2 == 0:
            return pos
        else:
            return 9 + pos

  def decode_action(self, action):
      if self.game_type == "A":
        return self.letters_to_move[action]
      if self.game_type == "B":
        return self.letters_to_move[action % 9]

  def inform_action(self, state, player_id, action):
    """Let the bot know of the other agent's actions."""
    self.cur_state += self.decode_action(action)
    

  def step(self, state):
    """Returns the selected action and steps the internal state forward."""
    is_max = (state.current_player() == 0)
    next_move = self.determine_move(self.cur_state, is_max)
    next_action = self.encode_action(next_move)
    self.cur_state += next_move
    return next_action


  def restart(self):
    self.cur_state = ''

  def restart_at(self, state):
    self.restart()
    new_state = self._game.new_initial_state()
    for action in state.history():
      self.inform_action(new_state, new_state.current_player(),
                         new_state.action_to_string(action))
      new_state.apply_action(action)

#   @Memoize_tree   # Decorate the minimax algorithm
#   def minmax_tt(self, current_id, is_max):
#     tree = self.tree
#     current_node = tree[current_id] 
#     if current_node.data.is_endstate():
#         return current_node.data.get_value()
#     children_of_current_id = tree.children(current_id)
#     scores = [self.minmax_tt(child.identifier, not is_max) for child in children_of_current_id]
#     if is_max:
#         return max(scores)
#     else:
#         return min(scores)
  
#   def minmax_tt(self, current_id, is_max):
#     tree = self.tree
#     current_node = tree[current_id]                     # Find the tree element we are now
#     if current_node.data.is_endstate():                 # Are we at the end of the game?
#         return current_node.data.get_value()            # Return the value
#     children_of_current_id = tree.children(current_id)  # Determine the children
#     scores = [self.minmax_tt(child.identifier, not is_max) for child in children_of_current_id]   # Recursively run this function on each of the children
#     if is_max:                                          # Return the max or min score depending on which player we are
#         return max(scores)
#     else:
#         return min(scores)

  def determine_move(self, current_id, is_max):
    '''
    Given a state on the board, what is the best next move? 
    '''
    tree = self.tree
    potential_moves = tree.children(current_id)
    moves = [child.identifier[-1] for child in potential_moves]
    raw_scores = [minmax_tt(tree, child.identifier, not is_max) for child in potential_moves]
    if is_max:
        return moves[raw_scores.index(max(raw_scores))]
    else:
        return moves[raw_scores.index(min(raw_scores))]

  def who_can_win(self):
    if len(self.cur_state) % 2 == 0:
        if (minmax_tt(self.tree, self.cur_state, True) > 0):
            return 0
        if (minmax_tt(self.tree, self.cur_state, True) < 0):
            return 1
    else:
        if(minmax_tt(self.tree, self.cur_state, False) < 0):
            return 1
        if(minmax_tt(self.tree, self.cur_state, False) > 0):
            return 0

    return None

  def warmup_cache(self):
    all_states = []
    for length in range(1,9):
        tree_states = [''.join(state) for state in list(itertools.permutations(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], r=length))]
        all_states.extend(tree_states)

    for state in tqdm(all_states):
        try:
            move = self.determine_move(state, False) 
        except: # Skip any board states that cannot occur
            pass




