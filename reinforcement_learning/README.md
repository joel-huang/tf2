# Deep RL using Advantage Actor-Critic (A2C) Methods
Special thanks to this [TensorFlow 2.0 DRL guide](http://inoryy.com/post/tensorflow2-deep-reinforcement-learning).

## Actor-Critic Basics
Our network provides estimates of the optimal Policy, and the Value.
![](https://i.imgur.com/CL0w8rl.png "Network")

* Value methods reduce the error of the expected state-action values.
* Policy Gradient methods optimize the policy (via estimation) by adjusting its parameters.
* The most popular approach is a hybrid of the two: actor-critic methods, where the agents' policy is optimized through policy gradients, while a value based method is used as a bootstrap for the expected value estimates.

## Advantage Actor-Critic Methods
We improve stability and sample efficiency by adding the following improvements:

1. Policy gradients are weighted with returns (discounted future rewards), which somewhat alleviates the credit assignment problem, and resolves theoretical issues with infinite timesteps.

2. An advantage function is used instead of raw returns. Advantage is formed as the difference between returns and some baseline (e.g. state-action estimate) and can be thought of as a measure of how good a given action is compared to some average.

3. An additional entropy maximization term is used in the objective function to ensure agent sufficiently explores various policies. Entropy measures how random a probability distribution is - The uniform distribution maximizes this.
