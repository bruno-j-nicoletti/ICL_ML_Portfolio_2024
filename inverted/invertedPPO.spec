{
  "physicalModel": "invertedPendulum",
  "technique" : "ppo",
  "params": {
    "seed" : 10,
    "actor.learningRate": 0.001,
    "actor.discountFactor": 0.99,
    "actor.nHidden" : 2,
    "actor.hiddenSize" : 32,
    "actor.varianceScale" : 0.2,
    "actor.hiddenActivation" : "ELU",
    "InvertedPendulum/speedPenalty": 0.1,
    "targetKL" : 0.025
  }
}
