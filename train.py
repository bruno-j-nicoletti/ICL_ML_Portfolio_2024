#! /usr/bin/env python
from typing import Any, Optional
import argparse
from functools import partial
import random
import signal
import torch.nn as nn
import torch
import numpy as np

import GTS

gTheAgent: Optional[GTS.Agent] = None


def interrupt_handler(signum, frame):
    if gTheAgent:
        gTheAgent.abort()


def setSeed(seed: Any) -> None:
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


################################################################################
def main():
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train a neural net to animate a given gymnasium model.")
    parser.add_argument(
        "spec",
        type=str,
        help=
        "The file containing the training spec or a previously saved checkpoint to continue training."
    )
    parser.add_argument("output",
                        type=str,
                        help="Where to write out the result.")
    parser.add_argument("-s",
                        "--steps",
                        type=int,
                        default=200_000,
                        help="The number of steps to train.")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=50,
                        help="The number of epochs to train PPO for.")
    parser.add_argument("-c",
                        "--checkpoints",
                        type=str,
                        default="",
                        help="Where to write in progress checkpoints out to.")
    parser.add_argument("--checkinterval",
                        type=int,
                        default=100,
                        help="How often to write a check point.")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--nosaveoninterrupt', action='store_true')
    parser.add_argument(
        '-p',
        "--param",
        nargs="*",
        help="""override a param in the .spec file, eg -p whiteBaseline:true"""
    )

    args = parser.parse_args()

    try:
        # try it as a checkpoint
        agent = GTS.loadAgentFromFile(args.spec)
        modelSpec = agent.modelSpec()
        setSeed(agent.params().get("seed", 1))
    except:
        with open(args.spec, 'r') as file:
            trainingSpec = GTS.TrainingSpec.fromJSON(file.read())
        setSeed(trainingSpec.params.get("seed", 1))
        agent = GTS.makeAgentFromSpec(trainingSpec)

    modelSpec = agent.modelSpec()
    params = agent.params()

    logger: GTS.Logger | None = None
    if args.verbose:
        logger = GTS.Logger()

    global gTheAgent
    gTheAgent = agent
    signal.signal(signal.SIGINT, interrupt_handler)
    print(f"Training {args.spec} with {agent.agentName()}")
    agent.train(logger=logger,
                nSteps=args.steps,
                nEpochs=args.epochs,
                stepsPerEpoch=args.steps // args.epochs,
                checkpoints=args.checkpoints,
                checkpointInterval=args.checkinterval)
    testScores = GTS.test(agent.modelSpec(),
                          agent.policyNetwork(),
                          params,
                          50,
                          500,
                          deterministic=True)
    agent.setScore(testScores)
    print(f"SCORE IS {testScores}")

    if not agent.aborted() or not args.nosaveoninterrupt:
        agent.save(args.output)


if __name__ == "__main__":
    main()
