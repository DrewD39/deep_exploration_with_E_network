from lib.setting import *
from lib.model import selectNet
from lib.dataset import CartPoleVision
from lib.action_selection import epsilon_greedy, LLL_epsilon_greedy, softmax, LLL_softmax
from lib.training import Trainer, DoraTrainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DORA training")
    parser.add_argument('-m', '--mode', choices=['dqn_greedy', 'dqn_softmax', 'dora_greedy',\
                                                'dora_softmax', 'dqn_opt_greedy', 'dqn_opt_softmax'],
                        help='dqn, dora, optimistic dqn - epsilon greedy or softmax', default='dora_greedy')
    parser.add_argument('-n', '--name', help='name to save', default='default')
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args

def dqn_run(run_name='default', plot=False):
    selection = epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def dqn_run_softmax(run_name='default', plot=False):
    selection = softmax()
    env = CartPoleVision()
    Qnet = selectNet()

    t = Trainer(Qnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def dora_run(run_name='default', plot=False):
    selection = LLL_epsilon_greedy()
    env = CartPoleVision()
    Qnet = selectNet()
    Enet = selectNet(Enet=True)

    t = DoraTrainer(Qnet, Enet, env, selection, lr=0.01, run_name=run_name, plot=plot)
    t.run()

def dora_run_softmax(run_name='default', plot=False):
    selection = LLL_softmax()
    env = CartPoleVision()
    Qnet = selectNet()
    Enet = selectNet(Enet=True)

    t = DoraTrainer(Qnet, Enet, env, selection, lr=0.01, run_name=run_name, plot=plot)
    t.run()

def optimistic_dqn_run(run_name='default', plot=False):
    selection = epsilon_greedy()
    env = CartPoleVision()
    optQnet = selectNet(Mode="optimisticQ")

    t = Trainer(optQnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def optimistic_dqn_run_softmax(run_name='default', plot=False):
    selection = softmax()
    env = CartPoleVision()
    optQnet = selectNet(Mode="optimisticQ")

    t = Trainer(optQnet, env, selection, run_name=run_name, plot=plot)
    t.run()

def run(args):
    if args.mode == 'dqn_greedy':
        dqn_run(run_name=args.name, plot=args.plot)
    if args.mode == 'dqn_softmax':
        dqn_run_softmax(run_name=args.name, plot=args.plot)
    if args.mode == 'dora_greedy':
        dora_run(run_name=args.name, plot=args.plot)
    if args.mode == 'dora_softmax':
        dora_run_softmax(run_name=args.name, plot=args.plot)
    if args.mode == 'dqn_opt_greedy':
        optimistic_dqn_run(run_name=args.name, plot=args.plot)
    if args.mode == 'dqn_opt_softmax':
        optimistic_dqn_run_softmax(run_name=args.name, plot=args.plot)



if __name__ == '__main__':
    args = parse_args()
    run(args)
