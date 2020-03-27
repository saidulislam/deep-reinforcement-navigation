## Deep Reinforcement Learning Nanodegree (Udacity)
## Project 1: Navigation
#### Train a Reinforcement Learning agent with DQN extensions
For this project, I built a reinforcement learning (RL) agent that learns to navigate and collect yellow bananas in a large, square world. 
The world/environment is similar to [Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

<img src="images/banana.gif" align="top-left" alt="" title="RL Agent" />

#### State
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions.

#### Action
Four discrete actions are available:
- `0` move forward
- `1` move backward
- `2` turn left
- `3` turn right
