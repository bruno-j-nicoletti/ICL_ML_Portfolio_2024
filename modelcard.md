# Model Card for GTS Trained Models
This is a model card that covers 9 neural networks which control two different [MuJoCo Gymnasium](https://gymnasium.farama.org/environments/mujoco/) simulation environments. The models have no practical use beyond being somewhat interesting as the main purpose was to see if aesthetically pleasing simulations of articulated figures could be achieved by reinforcement learning. A model card is not really appropriate for what I did, but one was demanded for the project, so here it is.

## Model Details
The the Gymnasium environments controlled by the neural networks, and the trained networks that are my models are...
- [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/)
    - hopperA2C.checkpoint : successful control model
    - hopperPPO.checkpoint : example bad model
    - hopperReinforce.checkpoint  : example bad model
    - tripping.checkpoint : example bad model (comedy)
    - worming.checkpoint : example bad model (comedy)
- [Walker2d-v4](https://gymnasium.farama.org/environments/mujoco/walker2d/)
    - walker/walkerA2C.checkpoint : successful but comedic model
    - walker/walkerPPOBad.checkpoint : example bad model (comedy)

### Model Description
- **Developed by:** Bruno Nicoletti
- **Shared by [optional]:** Bruno Nicoletti
- **Model type:** Stochastic Neural Networks
- **License:** MIT

The models are all [torch]](https://pytorch.org/) stochastic neural networks that generate one of more real valued outputs, each of which is used to control an input to the relevant Gymnasium simulation environment. There as saved in checkpoint files, along with the optimiser used to train them, so allowing for continued training.

## Uses
There are no practical uses for any of these models. You can run the simulations for your amusement, especially the funny ones. The models were trained in an attempt to see how reinforcement learning could be applied to computer animation.

### Direct Use
After following installation instructions, the models can only be used via the [play.py](play.py) program that is in the repository along side this model card. To run any model, simply go...

```
./play.py hopper/hopperA2C.checkpoint
```

This will run the simulation live for multiple steps, or until the model fails by falling over. It can be looped multiple times with `-l NUM_LOOPS` or continuously with `-l 0`. To have the simulation continue after if it ‘fails’, use the `-k` option. For example...

```
./play.py walker/walkerPPOBad.checkpoint  -k
```


### Out-of-Scope Use
There are no out of scope uses, do what you will with them.

## Bias, Risks, and Limitations
The models have no sociotechnical biases or risks involved with them. They are extremely limited in scope, being only able to drive the specific Gymnasium simulation environments.

## Training Details

### Training Data
There is no training data and so no data card, as all the inputs to the models are simulation environments used to train reinforcement learning agents. Each environment represents some figure that is simulated with a physics engine. At each time step, an agent applies forces or torques to some section of the figure, Gymnasium simulates the consequences of those forces and updates the figure. A ‘reward’ is returned as a consequence of that action, which is used to train the agent.

Each environment has a number of states that can be observed, and a number of actions (ie applied forces) that can be taken. These are all real valued numbers where the actions are bounded, but not the observed states.

#### [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/)
A single jointed leg, the observations are...

![Hopper](images/hopper.gif)

| Num | Observation                                        | Min  | Max |  Unit                     |
| --- | -------------------------------------------------- | ---- | --- |  ------------------------ |
| 0   | z-coordinate of the torso (height of hopper)       | -Inf | Inf |  position (m)             |
| 1   | angle of the torso                                 | -Inf | Inf |  angle (rad)              |
| 2   | angle of the thigh joint                           | -Inf | Inf |  angle (rad)              |
| 3   | angle of the leg joint                             | -Inf | Inf |  angle (rad)              |
| 4   | angle of the foot joint                            | -Inf | Inf |  angle (rad)              |
| 5   | velocity of the x-coordinate of the torso          | -Inf | Inf |  velocity (m/s)           |
| 6   | velocity of the z-coordinate (height) of the torso | -Inf | Inf |  velocity (m/s)           |
| 7   | angular velocity of the angle of the torso         | -Inf | Inf |  angular velocity (rad/s) |
| 8   | angular velocity of the thigh hinge                | -Inf | Inf |  angular velocity (rad/s) |
| 9   | angular velocity of the leg hinge                  | -Inf | Inf |  angular velocity (rad/s) |
| 10  | angular velocity of the foot hinge                 | -Inf | Inf |  angular velocity (rad/s) |


The available actions are...

| Num | Action                             | Control Min | Control Max | Unit         |
|-----|------------------------------------|-------------|-------------|--------------|
| 0   | Torque applied on the thigh rotor  | -1          | 1           | torque (N m) |
| 1   | Torque applied on the leg rotor    | -1          | 1           | torque (N m) |
| 2   | Torque applied on the foot rotor   | -1          | 1           | torque (N m) |


Rewards are 1 for each timestep the hopper stays upright, with a bonus for moving in the x-direction and penalty if the torques are too large.

#### [Walker2d-v4](https://gymnasium.farama.org/environments/mujoco/walker2d/)
A pair of legs, the observations are...

![Walker](images/walker2d.gif)

| Num | Observation                                        | Min  | Max | Unit                     |
| --- | -------------------------------------------------- | ---- | --- | ------------------------ |
| excluded | x-coordinate of the torso                     | -Inf | Inf | position (m)             |
| 0   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | position (m)             |
| 1   | angle of the torso                                 | -Inf | Inf | angle (rad)              |
| 2   | angle of the thigh joint                           | -Inf | Inf | angle (rad)              |
| 3   | angle of the leg joint                             | -Inf | Inf | angle (rad)              |
| 4   | angle of the foot joint                            | -Inf | Inf | angle (rad)              |
| 5   | angle of the left thigh joint                      | -Inf | Inf | angle (rad)              |
| 6   | angle of the left leg joint                        | -Inf | Inf | angle (rad)              |
| 7   | angle of the left foot joint                       | -Inf | Inf | angle (rad)              |
| 8   | velocity of the x-coordinate of the torso          | -Inf | Inf | velocity (m/s)           |
| 9   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | velocity (m/s)           |
| 10  | angular velocity of the angle of the torso         | -Inf | Inf | angular velocity (rad/s) |
| 11  | angular velocity of the thigh hinge                | -Inf | Inf | angular velocity (rad/s) |
| 12  | angular velocity of the leg hinge                  | -Inf | Inf | angular velocity (rad/s) |
| 13  | angular velocity of the foot hinge                 | -Inf | Inf | angular velocity (rad/s) |
| 14  | angular velocity of the thigh hinge                | -Inf | Inf | angular velocity (rad/s) |
| 15  | angular velocity of the leg hinge                  | -Inf | Inf | angular velocity (rad/s) |
| 16  | angular velocity of the foot hinge                 | -Inf | Inf | angular velocity (rad/s) |

The available actions are...

| Num | Action                                 | Control Min | Control Max | Unit         |
|-----|----------------------------------------|-------------|-------------|--------------|
| 0   | Torque applied on the thigh rotor      | -1          | 1           | torque (N m) |
| 1   | Torque applied on the leg rotor        | -1          | 1           | torque (N m) |
| 2   | Torque applied on the foot rotor       | -1          | 1           | torque (N m) |
| 3   | Torque applied on the left thigh rotor | -1          | 1           | torque (N m) |
| 4   | Torque applied on the left leg rotor   | -1          | 1           | torque (N m) |
| 5   | Torque applied on the left foot rotor  | -1          | 1           | torque (N m) |


Rewards are 1 for each timestep the walker stays upright, with a bonus for moving in the x-direction and penalty if the torques are too large.

### Training Procedure
The neural networks were trained via reinforcement learning, using one of three different RL agents. In all cases the basic procedure is the same, a neural network was defined whose input is the observations of the simulation environment, it then computes a value for each action needed by the environment.

The simulation is run for a number of time steps, at each step the neural network generates actions from observations, those actions are applied to the simulation. A reward is returned by the environment depending on how that action affected the environment. The rewards, action, and states are collected into an episode. At the end of an episode a reinforcement learning algorithm is run to train the neural network. This is repeated multiple times to train the network.

Three RL agents were used...
- REINFORCE with whitening baseline
    - I couldn’t get either the walker or hopper to succeed with this agent,
- Advantage Actor Critic (A2C)
    - which worked best of all the agents
- Proximal Policy Optimisation (PPO)
    - which I produced somewhat dubious results on everything but the inverted pendulum.

In all cases, the neural network used to compute the actions was a stochastic neural network, during training a value would be drawn from one distribution per action to allow for exploration. REINFORCE learns the standard deviation of the distribution as it goes, while A2C and PPO have a hyper parameter that specifies the standard deviation as a proportion of the range of a control.

#### Training Hyperparameters
Hyperparameter were explored using the [optuna](https://optuna.org/) parameter search framework, which I wrapped up in the [paramSearch.py](paramSearch.py) program. A training space file (eg: [hopper/hopperA2C.space](hopper/hopperA2C.space)) is passed to paramSearch.py and it will automate trials across multiple CPUs. Trials were scored by having the trained agent run 100 episodes of 1000 steps on a randomised environment, and computing the following...

$$mean = \frac{\sum trials}{|trials|}$$

$$cov = \frac{\sigma(trials)}{mean}$$

$$score = mean * (1 - cov)$$

I ran multiple passes over the hyper parameter spaces for each figure and each agent, choosing both both the [Tree-structured Parzen Estimator](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler) and the [Random](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler) samplers. I used the TPE sampler to find appropriate learning rates, net sizes and a few other parameterisations for all agents on all Gymnasium environments.

A significant problem was that the agents training the hopper and the walker could find local minima where they both stood still. Not at all aesthetically pleasing. They scored highly as they were being rewarded for not falling over for many timesteps. Attempting to move made falling over more likely, so even if they were reward for the motion, they were punished for falling, leading to a low overall score. Hyperparameterisations exist that let them move, but they are much less common. All this made the hyper parameter search hone in on the wrong thing, ie: hoppers and walkers that stood still.

The parameter that mattered most for getting the agents to actually move was the `actor.varianceScale` hyper param. This set the size of the stochastic distribution used by for PPO and A2C as a proportion of and action’s valid range. Too big it went crazy, too small and it didn’t move.

To get around this, the first approach I used was to a random search for the size of the distribution. I’d then play the higher scoring models, looking for networks that actually hopped or walked. I was starting to automate this by modifying the scoring system so that total rewards for a run were modulated by a function that measured how much overall motion there was at the end. It seemed to be a workable approach. But I stopped as I had to write up this model card instead.

To see the specific hyper parameters used within a given model, use the [info.py](info.py) program. For example...

```
> ./info.py hopper/hopperA2C.checkpoint
File: hopper/hopperA2C.checkpoint
model : hopper
trainer : A2C
Score
Params...
   nEnvironments : 8
   actor.varianceScale : 0.5038296102975216
   actor.varianceScaleDelta : 0.0
   actor.learningRate : 0.001
   actor.nHidden : 2
   actor.hiddenSize : 128
   actor.hiddenActivation : ReLU
   critic.learningRate : 0.005
   critic.nHidden : 2
   critic.hiddenSize : 64
   critic.hiddenActivation : ReLU
   lambda : 0.97
   discountFactor : 0.99
   maxEpisodeLength : 1000
   entropyCoeff : 0.01
   seed : 3
   actor.discountFactor : 0.99

```

Additionally, the total number of training steps are passed in as an argument, as were the number of epochs to train over for PPO and A2C. I should have formalised those into the automated hyper parameter exploration system as there are nuances as to what you should pass to each agent.

##### Reinforce
My implementation of Reinforce has the following hyper parameters.
  - policy.discountFactor : discount factor for rewards,
  - policy.nHidden : number of hidden layers used,
  - policy.hiddenSize : size of the hidden layers,
  - policy.hiddenActivation : activation function “ReLU”, “ELU” and more
  - policy.learningRate : learning rate
  - baseline :  "none" or "white"
  - seed : random seed to use

##### Advance Actor Critic 
My A2C implementation runs multiple simulation environments in parallel which are then all used to train the same neural network, as this can reduce variance markedly. The hyper parameters are...
- nEnvironments : the number of environments to train over
- actor.varianceScale : factor to multiply an action’s range by when setting the variance of the it’s distribution in a stochastic network
- actor.varianceScaleDelta : unused 
- actor.learningRate : actor learning rate
- actor.nHidden : number of hidden layers for the actor
- actor.hiddenSize : size of each of the actor’s hidden layers
- actor.hiddenActivation : activation function for the hidden layer
- critic.learningRate : critic learning rate
- critic.nHidden : number of hidden layers for the critic
- critic.hiddenSize : size of each of the critic’s hidden layers
- critic.hiddenActivation : activation function for the hidden layer
- lambda : lamdba factor used to discount advantage
- discountFactor : discount factor applied to rewards
- maxEpisodeLength : maximum episode length
- entropyCoeff : coefficient used to modulate loss due to entropy
- seed : random number seed

##### PPO
The hyper parameters are...
- actor.varianceScale : factor to multiply an action’s range by when setting the variance of the it’s distribution in a stochastic network- actor.learningRate : actor learning rate
- actor.nHidden : number of hidden layers for the actor
- actor.hiddenSize : size of each of the actor’s hidden layers
- actor.trainingIterations : how many times to train the actor on each pass
- actor.hiddenActivation : activation function for the hidden layer
- critic.learningRate : critic learning rate
- critic.nHidden : number of hidden layers for the critic
- critic.hiddenSize : size of each of the critic’s hidden layers
- critic.hiddenActivation : activation function for the hidden layer
- critic.trainingIterations : how many times to train the critic on each pass
- lambda : lambda factor used to discount advantages
- clipRatio : used by PPO to limit the loss function between training passes
- discountFactor : discount factor applied to rewards
- maxEpisodeLength : maximum episode length
- targetKL : max threshold for divergence between successive passes of the actor training
- seed : random number seed

## Evaluation
The primary criteria for evaluation was if the models move in a convincing way, the secondary criteria was that if they didn’t succeed, was it funny. 

### Results
I succeed in having the hopper hop convincingly via the A2C agent, it is saved in `hopper/hopperA2C.checkpoint`. Ironically my most successful attempt at getting the walker to move and not fall over had it hopping on one leg, this is in `walker/walkerA2C.checkpoint`. Several comedy agents are available in other checkpoints, be sure to play then with the `-k` option. They are aesthetically pleasing in a different way than originally intended.

#### Software
Implemented in python, using a range of third party libraries, most importantly PyTorch, and gymnasium.

## Model Card Contact
https://github.com/bruno-j-nicoletti