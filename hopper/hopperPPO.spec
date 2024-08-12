{
  "physicalModel": "hopper",
  "technique": "ppo",
  "params": {
    "actor.learningRate": 7.5e-05,
    "actor.nHidden": 2,
    "actor.hiddenSize": 256,
    "actor.hiddenActivation": "leakyReLU",
    "actor.varianceScale": 0.38464785150072,
    "actor.trainingIterations": 80,
    "critic.learningRate": 0.0003455968380598673,
    "critic.nHidden": 2,
    "critic.hiddenSize": 128,
    "critic.hiddenActivation": "leakyReLU",
    "critic.trainingIterations": 80,
    "clipRatio": 0.2,
    "targetKL": 0.025,
    "lambda": 0.97,
    "discountFactor": 0.99,
    "maxEpisodeLength": 1000,
    "seed": 34,
    "comment": "Fixing size of networks and activations from last pass.",
    "actor.discountFactor": 0.995
  }
}