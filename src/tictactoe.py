# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import colorama
from colorama import Fore, Back, Style

import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn

# Board size
board_size = 3

# Number of contiguous marks to win
marks_win = 3

# Win reward
REWARD_WIN = 1.
# Draw reward
REWARD_DRAW = 0.
# Ordinary action reward
REWARD_ACTION = 0.

# Reward discount rate
gamma = 0.8

# Initial exploration rate
epsilon_initial = 1.0
# Final exploration rate
epsilon_final = .01
# Number of episodes to anneal epsilon
epsilon_anneal_episodes = 5000

# Learning rate
learning_rate = .001

# Number of episodes to run
episode_max = 10000

# Number of episodes to accumulate stats
episode_stats = 100

# Run name for tensorboard
run_name = "%s" % int(time.time())

# Directory for storing tensorboard summaries
summary_dir = '/tmp/tensorflow/tictactoe'

def dump_board(sx, so, move_index=None, win_indices=None, q=None):
    """
    Dump board state to the terminal.
    """
    for i in xrange(board_size):
        for j in xrange(board_size):
            if (i, j) == move_index:
                color = Fore.GREEN
            else:
                color = Fore.BLACK
            if not win_indices is None and (i, j) in win_indices:
                color += Back.LIGHTYELLOW_EX
            print(" ", end="")
            if sx[i, j] and so[i, j]:
                print(Fore.RED + "?" + Fore.RESET, end="")
            elif sx[i, j]:
                print(color + "X" + Style.RESET_ALL, end="")
            elif so[i, j]:
                print(color + "O" + Style.RESET_ALL, end="")
            else:
                print(".", end="")
        if not q is None:
            print("   ", end="")
            for j in xrange(board_size):
                if (i, j) == move_index:
                    color = Fore.GREEN
                else:
                    color = Fore.BLACK
                if not (sx[i, j] or so[i, j]) or (i, j) == move_index:
                    print(color + " %6.3f" % q[i, j] + Style.RESET_ALL, end="")
                else:
                    print(Fore.LIGHTBLACK_EX + "    *  " + Style.RESET_ALL, end="")
        print()
    print()

def check_win(s):
    """
    Count marks and check for the 'win' state.
    """
    # Check rows and columns
    for i in xrange(board_size):
        marks_r = 0
        marks_c = 0
        win_indices_c = []
        win_indices_r = []
        for j in xrange(board_size):
            # Check row
            if s[i, j]:
                marks_r += 1
                win_indices_r.append((i, j))
                if marks_r >= marks_win:
                    return True, win_indices_r
            else:
                marks_r = 0
                win_indices_r = []

            # Check column
            if s[j, i]:
                marks_c += 1
                win_indices_c.append((j, i))
                if marks_c >= marks_win:
                    return True, win_indices_c
            else:
                marks_c = 0
                win_indices_c = []

    # Check diagonals
    for i in xrange(board_size):
        if i+1 < marks_win:
            continue # diagonals are shorter than number of marks to win
        marks_d1 = 0
        marks_d2 = 0
        marks_d3 = 0
        marks_d4 = 0
        win_indices_d1 = []
        win_indices_d2 = []
        win_indices_d3 = []
        win_indices_d4 = []
        for j in xrange(i+1):
            # Check first (top) pair of diagonals
            if s[j, i-j]:
                marks_d1 += 1
                win_indices_d1.append((j, i-j))
                if marks_d1 >= marks_win:
                    return True, win_indices_d1
            else:
                marks_d1 = 0
                win_indices_d1 = []

            if s[j, board_size-i+j-1]:
                marks_d2 += 1
                win_indices_d2.append((j, board_size-i+j-1))
                if marks_d2 >= marks_win:
                    return True, win_indices_d2
            else:
                marks_d2 = 0
                win_indices_d2 = []

            # Check second (bottom) pair of diagonals
            if i == board_size-1:
                continue # main diagonals already checked
            if s[board_size-i+j-1, j]:
                marks_d3 += 1
                win_indices_d3.append((board_size-i+j-1, j))
                if marks_d3 >= marks_win:
                    return True, win_indices_d3
            else:
                marks_d3 = 0
                win_indices_d3 = []

            if s[board_size-i+j-1, board_size-j-1]:
                marks_d4 += 1
                win_indices_d4.append((board_size-i+j-1, board_size-j-1))
                if marks_d4 >= marks_win:
                    return True, win_indices_d4
            else:
                marks_d4 = 0
                win_indices_d4 = []

    return False, []

def check_draw(sx, so):
    """
    Check for draw.
    """
    return np.all(sx+so)

def train(session, graph_ops, summary_ops, saver):
    """
    Train model.
    """
    # Initialize variables
    session.run(tf.initialize_all_variables())

    # Initialize summaries writer for tensorflow
    writer = tf.train.SummaryWriter(summary_dir + "/" + run_name, session.graph)
    summary_op = tf.merge_all_summaries()

    # Unpack graph ops
    q_nn, q_nn_update, s, a, y = graph_ops

    # Unpack summary ops
    win_rate_summary, episode_length_summary, epsilon_summary = summary_ops

    # Setup exploration rate parameters
    epsilon = epsilon_initial
    epsilon_step = (epsilon_initial - epsilon_final) / epsilon_anneal_episodes

    # X player state
    sx_t = np.empty([board_size, board_size], dtype=np.bool)
    # O player state
    so_t = np.empty_like(sx_t)

    # Accumulated stats
    stats = []

    # X move first
    move_x = True

    episode_num = 1

    while episode_num <= episode_max:
        # Start new game training episode
        sx_t[:] = False
        so_t[:] = False

        sar_prev = [(None, None, None), (None, None, None)] # [(s, a, r(a)), (s(a), o, r(o)]

        move_num = 1

        while True:
            # Observe the next state
            s_t = create_state(move_x, sx_t, so_t)
            # Get Q values for all actions
            q_t = q_values(s_t, session, q_nn, s)
            # Choose action based on epsilon-greedy policy
            q_max_index, a_t_index = choose_action(q_t, sx_t, so_t, epsilon)

            # Retrieve previous player state/action/reward (if present)
            s_t_prev, a_t_prev, r_t_prev = sar_prev.pop(0)

            if not s_t_prev is None:
                # Calculate updated Q value
                y_t_prev = r_t_prev + gamma * q_t[q_max_index]
                # Update Q network
                session.run(q_nn_update, feed_dict={s: [s_t_prev], a: [a_t_prev], y: [y_t_prev]})

            # Apply action to state
            r_t, sx_t, so_t, terminal = apply_action(move_x, sx_t, so_t, a_t_index)

            a_t = np.zeros_like(sx_t, dtype=np.float32)
            a_t[a_t_index] = 1.

            if terminal: # win or draw
                y_t = r_t # reward for current player
                s_t_prev, a_t_prev, r_t_prev = sar_prev[-1] # previous opponent state/action/reward
                y_t_prev = r_t_prev - gamma * r_t # discounted negative reward for opponent
                # Update Q network
                session.run(q_nn_update, feed_dict={s: [s_t, s_t_prev], a: [a_t, a_t_prev], y: [y_t, y_t_prev]})

                # Play test game before next episode
                length, win_x, win_o = test(session, q_nn, s)
                stats.append([win_x or win_o, length])
                break

            # Store state, action and its reward
            sar_prev.append((s_t, a_t, r_t))

            # Next move
            move_x = not move_x
            move_num += 1

        # Scale down epsilon after episode
        if epsilon > epsilon_final:
            epsilon -= epsilon_step

        # Process stats
        if len(stats) >= episode_stats:
            win_rate, length = np.mean(stats, axis=0)
            print("episode: %d," % episode_num, "epsilon: %.5f," % epsilon, \
                  "win rate: %.3f," % win_rate, "length: %.3f" % length)
            summary_str = session.run(summary_op, feed_dict={win_rate_summary: win_rate, \
                                                             episode_length_summary: length,
                                                             epsilon_summary: epsilon})
            writer.add_summary(summary_str, episode_num)
            stats = []

        # Next episode
        episode_num += 1

    test(session, q_nn, s, dump=True)

def test(session, q_nn, s, dump=False):
    """
    Play test game.
    """
    # X player state
    sx_t = np.zeros([board_size, board_size], dtype=np.bool)
    # O player state
    so_t = np.zeros_like(sx_t)

    move_x = True
    move_num = 1

    if dump:
        print()

    while True:
        # Choose action
        s_t = create_state(move_x, sx_t, so_t)
        # Get Q values for all actions
        q_t = q_values(s_t, session, q_nn, s)
        _q_max_index, a_t_index = choose_action(q_t, sx_t, so_t, -1.)

        # Apply action to state
        r_t, sx_t, so_t, terminal = apply_action(move_x, sx_t, so_t, a_t_index)

        if dump:
            if terminal:
                if move_x:
                    _win, win_indices = check_win(sx_t)
                else:
                    _win, win_indices = check_win(so_t)
            else:
                win_indices = None
            print(Fore.CYAN + "Move:", move_num, Fore.RESET + "\n")
            dump_board(sx_t, so_t, a_t_index, win_indices, q_t)

        if terminal:
            if not r_t:
                # Draw
                if dump:
                    print("Draw!\n")
                return move_num, False, False
            elif move_x:
                # X wins
                if dump:
                    print("X wins!\n")
                return move_num, True, False
            # O wins
            if dump:
                print("O wins!\n")
            return move_num, False, True

        move_x = not move_x
        move_num += 1

def q_values(s_t, session, q_nn, s):
    """
    Get Q values for actions from network for given state.
    """
    return q_nn.eval(session=session, feed_dict={s: [s_t]})[0]

def create_state(move_x, sx_t, so_t):
    """
    Create full state.
    """
    s_t = np.empty([2, board_size, board_size])

    if move_x:
        s_t[0] = sx_t
        s_t[1] = so_t
    else:
        s_t[0] = so_t
        s_t[1] = sx_t

    return s_t

def choose_action(q_t, sx_t, so_t, epsilon):
    """
    Choose action index for given state.
    """
    # Get valid action indices
    a_t_vindices = np.where((sx_t+so_t)==False)
    a_t_tvindices = np.transpose(a_t_vindices)

    q_max_index = tuple(a_t_tvindices[np.argmax(q_t[a_t_vindices])])

    # Choose next action based on epsilon-greedy policy
    if np.random.random() <= epsilon:
        # Choose random action from list of valid actions
        a_t_index = tuple(a_t_tvindices[np.random.randint(len(a_t_tvindices))])
    else:
        # Choose valid action w/ max Q
        a_t_index = q_max_index

    return q_max_index, a_t_index

def apply_action(move_x, sx, so, a_index):
    """
    Apply action to state, get reward and check for terminal state.
    """
    if move_x:
        s = sx
    else:
        s = so
    s[a_index] = True
    win, _win_indices = check_win(s)
    if win:
        return REWARD_WIN, sx, so, True
    if check_draw(sx, so):
        return REWARD_DRAW, sx, so, True
    return REWARD_ACTION, sx, so, False

def build_summaries():
    win_rate_op = tf.Variable(0.)
    tf.scalar_summary("Win Rate", win_rate_op)
    episode_length_op = tf.Variable(0.)
    tf.scalar_summary("Episode Length", episode_length_op)
    epsilon_op = tf.Variable(0.)
    tf.scalar_summary("Epsilon", epsilon_op)
    return win_rate_op, episode_length_op, epsilon_op

def build_graph():
    s = tf.placeholder(tf.float32, [None, 2, board_size, board_size], name="s")

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(s, [0, 2, 3, 1])

    # Flatten inputs
    net = tf.reshape(net, [-1, int(np.prod(net.get_shape().as_list()[1:]))])

    # Hidden fully connected layer
    net = layers.fully_connected(net, 50, activation_fn=nn.relu)

    # Output layer
    net = layers.fully_connected(net, board_size*board_size, activation_fn=None)

    # Reshape output to board actions
    q_nn = tf.reshape(net, [-1, board_size, board_size])

    # Define loss and gradient update ops
    a = tf.placeholder(tf.float32, [None, board_size, board_size], name="a")
    y = tf.placeholder(tf.float32, [None], name="y")
    action_q_values = tf.reduce_sum(tf.mul(q_nn, a), reduction_indices=[1, 2])
    loss = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    q_nn_update = optimizer.minimize(loss, var_list=tf.trainable_variables())

    return q_nn, q_nn_update, s, a, y

def main(_):
    with tf.Session() as session:
        graph_ops = build_graph()
        summary_ops = build_summaries()
        saver = tf.train.Saver(max_to_keep=5)
        train(session, graph_ops, summary_ops, saver)

if __name__ == "__main__":
    colorama.init()
    flags = tf.app.flags
    flags.DEFINE_string("name", run_name, "Run name")
    FLAGS = flags.FLAGS
    run_name = FLAGS.name
    tf.app.run()
