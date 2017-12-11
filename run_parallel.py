from utility.parallel_run import map_parallel
from main import run
import copy

class Args:
    def __init__(self, mode='dora', name='default', plot=False):
        self.mode = mode
        self.name = name
        self.plot = plot

if __name__ == "__main__":

    tasks = []

    ## Run for each of the 6 modes
    for mode in ['dqn_greedy', 'dqn_softmax', 'dora_greedy',\
                'dora_softmax', 'dqn_opt_greedy', 'dqn_opt_softmax']:

        ## run each mode 10 times and save results with different names
        nameList = map(str, range(10));
        nameList = map((lambda x: mode + "_" + x), nameList)
        #for i in range(nameList):
        #    nameList[i] = mode + "_" + nameList[i]
        for name in nameList:
            plot = False
            tasks.append(Args(mode, name, plot))

    map_parallel(run, tasks)
