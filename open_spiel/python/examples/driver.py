from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import sys

from absl import app
from absl import flags
import numpy as np

from multiprocessing import Pool

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
from open_spiel.python.games.tic_tac_toe import TicTacToeGame

FLAGS = flags.FLAGS

flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 1000, "How many simulations to run.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_string("game_type", None, "TTT Game type")

_NUM_COLS = 3 

def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)

def convert_str_board_state(string, state):
    points = str(string).strip()
    points = points.split(',')
    for idx, point in enumerate(points):
        if int(point) == 1:
            state[_coord(idx)] = 'x'
        elif int(point) == 0:
            state[_coord(idx)] = '.'
        elif int(point) == -1:
            state[_coord(idx)] = 'o'
        else:
            raise Exception("Invalid cell number")

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

def worker(flags, game_type, states):
    rng = np.random.RandomState(flags['seed'])
    model = az_model.Model.from_checkpoint(flags['az_path'])

    game = TicTacToeGame(game_type=game_type)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    bot = mcts.MCTSBot(
        game,
        flags['uct_c'],
        flags['max_simulations'],
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=flags['solve'],
        verbose=flags['verbose'])
    
    total_lines = 0
    total_valid_moves = 0

    
    for line in states:
        state = game.new_initial_state()
        current_state_str = line.strip()
        convert_str_board_state(current_state_str, state.board)
        num_xs = [a for a in current_state_str.split(',') if a == '1']
        num_os = [a for a in current_state_str.split(',') if a == '-1']
        if num_xs == num_os:
            state._cur_player = 0
        else:
            state._cur_player = 1

        action = bot.step(state)
        action_str = state.action_to_string(0, action)
        state.apply_action(action)
        result_state_str = convert_board_state_str(state.board)
        # print('Current state', current_state_str)
        # print('Next state', result_state_str)
        total_lines += 1
        if is_valid_move(current_state_str, result_state_str):
            total_valid_moves += 1
        # else:
            # print('Illegal move', (current_state_str, result_state_str))
        print(total_lines)
        bot.restart()
        # if total_lines > 100:
        #     break
        # print('Is valid move', is_valid_move(current_state_str, result_state_str))
        # print('Action performed', action_str)
    
    return (total_valid_moves, total_lines)


def main(argv):
    num_workers = 20
    game_type = FLAGS.game_type
    valid_states = []
    with open('/home/midhul/oracle_final.txt', 'r') as positive_file:
        for line in positive_file:
            if line.strip() == '':
                continue
            valid_states.append(line.strip())

    print('Total valid states')
    print(len(valid_states))

    shards = []
    for wid in range(num_workers):
        shard = []
        for i in range(len(valid_states)):
            if i % num_workers == wid:
                shard.append(valid_states[i])
        shards.append(shard)

    params = {}
    params['seed'] = FLAGS.seed
    params['az_path'] = FLAGS.az_path
    params['uct_c'] = FLAGS.uct_c
    params['max_simulations'] = FLAGS.max_simulations
    params['solve'] = FLAGS.solve
    params['verbose'] = FLAGS.verbose


    results = []
    with Pool(num_workers) as p:
        results = p.starmap(worker, [(params, game_type, shards[i]) for i in range(num_workers)])
    
    total_valid_moves = 0
    total_lines = 0
    for res in results:
        total_valid_moves += res[0]
        total_lines += res[1]

    print('Total valid moves', total_valid_moves)
    print('Total states', total_lines)
    
if __name__ == '__main__':
  app.run(main)
