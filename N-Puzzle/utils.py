import random
import multiprocessing as mp
from multiprocessing import Manager, Pool
import environment
from agent import Agent


def do_parallel_processing_on_agent(N, initial_state_string, d, *args):
    env = environment.Environment(N)
    agent = Agent(env, d)
    agent.n_step_TD(initial_state_string, *args)


def get_random_state_string(n):
    
    lst = list(range(n * n))
    random.shuffle(lst)

    result = ""

    for i in range(n * n):
        tmp = str(lst[i])

        if len(tmp) == 1:
            result += "0" + tmp
        else:
            result += tmp

    return result


def parallel_processing(process_count, N, *args):

    initial_state_string = get_random_state_string(N)

    with Manager() as manager:
        
        d = manager.dict()
        
        with Pool(processes=10) as pool:
            pool.starmap(do_parallel_processing_on_agent, [(N, initial_state_string, d, *args) for _ in range(process_count)])

        # processes = []
        # for _ in range(process_count):
        #     p = mp.Process(target=do_parallel_processing_on_agent, args=(N, initial_state_string, d, *args))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()


        env = environment.Environment(N)
        agent = Agent(env, d)
        agent.exploit(initial_state_string)
