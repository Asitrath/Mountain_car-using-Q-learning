import gym
import numpy as np
import matplotlib.pyplot as plt


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def discretize_state(state, state_bounds, num_buckets): # discretize states
    discretized = list()
    for i in range(len(state)):
        scaling = (state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        new_state = int(round((num_buckets[i] - 1) * scaling))
        new_state = min(num_buckets[i] - 1, max(0, new_state))
        discretized.append(new_state)
    return tuple(discretized)

def choose_action(state, Q, epsilon, num_actions): # epsilon greedy
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state])

def update_q_table(Q, state, action, reward, next_state, gamma, alpha): # Q table
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error

def run_ep(env, num_episodes): # Q learning

    # Discretize the state space
    num_buckets = (20, 20)  # 20 intervals for position and 20 for velocity
    num_actions = env.action_space.n  # 3 possible actions

    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    # Define hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    min_epsilon = 0.01
    Q = np.zeros(num_buckets + (num_actions,))
    epsilon = 1.0
    successes = np.zeros(num_episodes)
    steps_per_episode = np.zeros(num_episodes)

    for episode in range(num_episodes):
        current_state = discretize_state(env.reset(), state_bounds, num_buckets)
        done = False
        steps = 0

        while not done:
            env.render()
            action = choose_action(current_state, Q, epsilon, num_actions)
            next_state_raw, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state_raw, state_bounds, num_buckets)
            update_q_table(Q, current_state, action, reward, next_state, gamma, alpha)
            current_state = next_state
            steps += 1

            if done and next_state_raw[0] >= 0.5:
                successes[episode] = 1
        
        steps_per_episode[episode] = steps

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # if (episode + 1) % 100 == 0:
        #     plot_value_function(Q, episode + 1)    

    cumulative_successes = np.cumsum(successes)
    return cumulative_successes, steps_per_episode

def plot_value_function(q,episode):
    value_function = np.max(q, axis=2)
    #print(value_function)
    plt.imshow(value_function, origin='lower')
    plt.colorbar()
    plt.title(f"Value Function after {episode} episodes")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()


def aggregation(env): # Run it 10 times
    num_episodes = 1000
    num_runs = 10
    all_successes = np.zeros((num_runs, num_episodes))
    all_steps_per_episode = np.zeros((num_runs, num_episodes))

    for run in range(num_runs):
        cumulative_successes, steps_per_episode = run_ep(env, num_episodes)
        all_successes[run] = cumulative_successes
        all_steps_per_episode[run] = steps_per_episode
        print(f"Run {run + 1}/{num_runs} completed")

    # Calculate averages
    avg_successes = np.mean(all_successes, axis=0)
    avg_steps_per_episode = np.mean(all_steps_per_episode, axis=0)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(range(num_episodes), avg_successes)
    ax1.set_title('Averaged Cumulative Number of Successes')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Successes')

    ax2.plot(range(num_episodes), avg_steps_per_episode)
    ax2.set_title('Averaged Number of Steps per Episode')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Steps per Episode')

    plt.tight_layout()
    plt.show()


def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    # random_episode(env)
    run_ep(env, num_episodes=600)
    # aggregation(env)
    env.close()


if __name__ == "__main__":
    main()