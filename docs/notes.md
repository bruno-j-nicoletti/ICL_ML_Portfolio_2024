## The Plan
Teaching one of the simulated skeletons from Gymnasium to move
  - using the MuJoCo physics simulation library
      - ant, hopper or swimmer
      - possibly tweaking one of the figures
  - limited compute so going for a fairly simple model

- Proximal Policy Optimisation
    - seems to be the thing
    - with generalise advantage estimation
    - hyper params
        - NN stuff
            - layers, batch size
            - tanh activation as I need real valued numbers for a probability distribution
        - $\epsilon$ for the PPO clip loss function


## 2024-07-18 Matilde Meeting
- need a couple of days for
    - model card
    - non technical report
- extended it
- competition deadline is same as for project
    - noisy functions
        - wide exploration all the time
        - can train reinforcement learning to remove noise as an initial pass
    - share everything with them
    - write up
        - reflection on what I did
            - what worked and what didn’t
        - not over complicated

## 2024-07-22 Physical Model
- want to be able to abstract the MuJoCo physical model for...
    - actions
        - we know the joints we can apply torques to
    - we know some key values for the cost function
        - velocity of the body
        - angular velocities
        - angles

## 2024-07-24 Progress
- have written
    - modules
        - physical model abstraction
            - can override reward function if need be
        - reinforce trainer
        - parametric NN
    - a trainer program
        - writes out a net
        - needs to write out a check point as well
    - a player program
        - to run the net in a live MuJoCo session
- need to look at
    - normalisation of the environment
    - for PPO
      - see https://github.com/ericyangyu/PPO-for-Beginners
- TODO
    - parameterise the physical model’s reward functions
        - eg: the amount to punish the size of forces applied

## 2024-07-25 Progress
- for reinforce using the sum of the log probs of the separate actions as the probability needs to be one value
- I can get Reinforce to work happily with the inverted pendulum, struggles with the hopper, manually playing with hyperparams gets me no where
    - will set up a simple quantised hyper param search for
        - size of hidden layers
            - 16, 32, 64, 128
        - learning rate
            - some range
        - gamma
            - some range

## 2024-07-26 Progress
- Wrote a grid search hyper parameter explorer
  - dispatch.py
  - does complete search over the specified space
  - works on quantised/categorical values
- ran inverted over night
  - very sensitive to seeds!
- the suggestion for this seems to be to use ensemble methods

## 2024-07-29 Progress
- fixed it so that it is deterministic for a given seed
- Changed the stochastic net so that it initialises the weights with an optimal method
- tidied up code in prep for PPO
    - broke up ModelRunner into distinct functions
- can now record videos from play.py
- saving checkpoints every 1000 episodes during training
- hopper really doesn’t want to converge!

## 2024-07-30 Progress
- had a bug with a python `match` statement where activation function for hidden were not actually being set
- fixed that but it broke everything, nothing converges
- gah!

## 2024-07-31 Progress
- I was being silly. The reason it worked before I added activation layer was that the linear combination of inputs was large enough to generate actions that worked-ish. Adding tanh was silly as it broke that because it clips outputs to [-1, 1], and the output layer was limited in scope as to the values it could use. ie: don’t use tanh for regression problems. Flipping to ELU to ReLU fixes all of that. Hurrah!
- I need to work on Reinforce with a baseline next, then PPO for my three methods
- need a better termination condition rather than running for N episodes
- need hyper param explorer that is not exhaustive

## 2024-08-05 Progress
- reinforce with whitened baseline working
- need to change specs so that I can specify value layers

## 2024-08-06 Progress
- can't get reinforce with learned baseline to behave at all
- burned old grid search and implemented multithreaded paramSearch.py based on optuna python package
  - had to change how params and param spaces are specified and how NNs and optimisers are created
  - works for inverted pendulum quite happily
- need to tweak the hopper reward so it moves!
- handling exceptions, nans and bad rewards failures

## 2024-08-08 Progress
- have PPO working-ish
- seems to get the hopper to move a bit better
- supporting walker2d and haldCheetah environments
- lots of fiddling with bits
- have run some param searches, will set some big batches off overnight for walker and hopper