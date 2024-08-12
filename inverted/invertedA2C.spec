{
  "physicalModel": "invertedPendulum",
  "technique" : "a2c",
  "params": {
    "seed" : 42,
    "actor.learningRate": 0.001,
    "actor.discountFactor": 0.999,
    "actor.nHidden" : 2,
    "actor.hiddenSize" : 32,
    "actor.hiddenActivation" : "ReLU",
    "actor.varianceScale" : 0.0833333,
    "critic.learningRate": 0.005,
    "critic.nHidden" : 2,
    "critic.hiddenSize" : 32,
    "critic.hiddenActivation" : "ReLU",
    "InvertedPendulum/speedPenalty": 0.1
  }
}
