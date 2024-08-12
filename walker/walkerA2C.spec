{
  "physicalModel": "walker",
  "technique": "a2c",
  "params": {
    "nEnvironments": 8,
    "actor.varianceScale": 0.45235299113766636,
    "actor.varianceScaleDelta": 0.0,
    "actor.learningRate": 0.0005168910655945735,
    "actor.nHidden": 2,
    "actor.hiddenSize": 128,
    "actor.hiddenActivation": "ReLU",
    "critic.learningRate": 0.0031937800065171026,
    "critic.nHidden": 2,
    "critic.hiddenSize": 64,
    "critic.hiddenActivation": "ReLU",
    "lambda": 0.97,
    "discountFactor": 0.99,
    "maxEpisodeLength": 1000,
    "entropyCoeff": 0.01,
    "seed": 1,
    "actor.discountFactor": 0.99
  }
}