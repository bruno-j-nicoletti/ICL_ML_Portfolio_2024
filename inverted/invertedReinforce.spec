{
  "physicalModel": "invertedPendulum",
  "technique" : "reinforce",
  "params": {
    "seed" : 10,
    "policy.learningRate": 0.001,
    "policy.discountFactor": 0.99,
    "policy.nHidden" : 2,
    "policy.hiddenSize" : 16,
    "policy.hiddenActivation" : "ELU",
    "baseline": "white",
    "baseline.nHidden" : 2,
    "baseline.hiddenSize" : 128,
    "baseline.hiddenActivation" : "ReLU",
    "baseline.learningRate": 0.0001,
    "InvertedPendulum/speedPenalty": 0.1
  }
}
