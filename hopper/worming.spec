{
  "physicalModel": "hopper",
  "technique": "ppo",
  "params": {
    "actor.learningRate": 0.00024057170086583238,
    "actor.nHidden": 2,
    "actor.hiddenSize": 177,
    "actor.hiddenActivation": "ELU",
    "actor.varianceScale": 0.2,
    "actor.trainingIterations": 80,
    "critic.learningRate": 0.0002455968380598673,
    "critic.nHidden": 2,
    "critic.hiddenSize": 159,
    "critic.hiddenActivation": "ELU",
    "critic.trainingIterations": 80,
    "clipRatio": 0.2,
    "targetKL": 0.015,
    "lambda": 0.97,
    "discountFactor": 0.99,
    "maxEpisodeLength": 1000,
    "seed": 1,
    "actor.discountFactor": 0.995
  }
}