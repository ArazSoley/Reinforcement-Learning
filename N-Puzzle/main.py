import utils


if __name__ == "__main__":
    N = 4
    n = 2500
    max_episodes = 250
    max_steps = 5000
    epsilon_start = 0.6
    epsilon_end = 0.001
    default_state_value = 0.0
    update_on_increase = True
    epsilon_decay_type = "linear"
    process_count = 3

    utils.parallel_processing(process_count, N, n, 
                                                max_episodes, 
                                                max_steps, 
                                                epsilon_start, 
                                                epsilon_end, 
                                                default_state_value, 
                                                update_on_increase, 
                                                epsilon_decay_type)

