{
  "physicalModel": "walker",
  "technique" : "ppo",
  "params": {
    "seed" : 10,
    "targetKL" : 0.015,
    "actor.learningRate": 0.0002,
    "actor.discountFactor": 0.99,
    "actor.nHidden" : 2,
    "actor.hiddenSize" : 128,
    "actor.varianceScale" : 0.45,
    "actor.hiddenActivation" : "ELU",
    "critic.learningRate": 0.0002,
    "critic.nHidden" : 2,
    "critic.hiddenSize" : 64,
    "critic.hiddenActivation" : "ELU"
  }
}
