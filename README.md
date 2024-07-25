# Mountain Car Q-Learning

This project implements Q-learning to solve the Mountain Car problem using OpenAI Gym.

![Recording 2024-07-25 at 12 27 11](https://github.com/user-attachments/assets/0a8ca0ad-925c-4bf2-a4b8-c23725bf09d5)

Key features:
- Q-learning implementation for the Mountain Car environment
- State space discretization
- Epsilon-greedy action selection
- Visualization of the learned value function
- Option to run multiple episodes and aggregate results

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenAI Gym

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/mountain-car-qlearning.git
cd mountain-car-qlearning
pip install numpy matplotlib gym
```

## Usage

Run the main script to execute the Q-learning algorithm:

```bash
python mountain_car_qlearning.py
```

## Implementation Details

- `discretize_state`: Discretizes the continuous state space
- `choose_action`: Implements epsilon-greedy action selection
- `update_q_table`: Updates the Q-table using the Q-learning update rule
- `run_ep`: Runs a single episode of Q-learning
- `plot_value_function`: Visualizes the learned value function
- `aggregation`: Runs multiple episodes and aggregates results

## Hyperparameters

- State discretization: 20 buckets for position and velocity
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Initial exploration rate (ε): 1.0
- Exploration decay rate: 0.995
- Minimum exploration rate: 0.01

## Results

The script can output:
- Plots of the value function at different stages of learning
- Graphs showing the average cumulative successes and steps per episode over multiple runs

## Customization

You can modify the `main` function to:
- Run a single episode with `run_ep`
- Perform multiple runs and aggregate results with `aggregation`
- Visualize a random episode with `random_episode`

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the Mountain Car environment from OpenAI Gym and implements Q-learning as described in reinforcement learning literature.
