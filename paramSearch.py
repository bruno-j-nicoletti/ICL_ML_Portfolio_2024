#! /usr/bin/env python
import argparse
import asyncio
import optuna
import numpy as np
import os
import random
import shutil
import signal
import time
import torch
from typing import List, Optional, Tuple

import GTS


def resolveStorage(storage: str) -> str:
    """If storage is a url return it, otherwise make it a sqlite db"""
    if "://" in storage:
        return storage
    return f"sqlite:///{storage}"


################################################################################
def minion(args: argparse.Namespace) -> None:
    """Run in minion mode, which runs trial, possibly as a sub process."""

    # load the training space
    with open(args.space, 'r') as file:
        trainingSpace = GTS.TrainingSpace.fromJSON(file.read())

    # get the optuna study
    study = optuna.load_study(study_name=trainingSpace.name,
                              storage=resolveStorage(args.storage))

    # gh
    modelSpec = GTS.fetchPhysicalModelSpec(trainingSpace.physicalModel)

    def objective(trial: optuna.trial.Trial) -> float:
        # the objective function optuna uses for scoring

        # bake out a training spec from the space of hyper params
        trainingSpec = trainingSpace.bake(trial)
        params = trainingSpec.params
        trial.set_user_attr("GTS_PARAMS", params)
        trial.set_user_attr("GTS_SPEC", trainingSpec.toJSON())

        # start logging
        logdir = f"{args.logdir}/{trainingSpace.name}"
        logger = GTS.Logger(logdir, f"trial_{trial.number}")

        try:
            seed = params.get("seed", 1)
            assert isinstance(seed, int)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # make an agent from the hyper params and train it
            agent = GTS.makeAgentFromSpec(trainingSpec)
            nSteps = trainingSpace.nSteps
            nEpochs = trainingSpace.nEpochs
            stepsPerEpoch = nSteps // trainingSpace.nEpochs
            agent.train(logger=logger,
                        nSteps=nSteps,
                        nEpochs=nEpochs,
                        stepsPerEpoch=stepsPerEpoch)

            # score the result
            testScores = GTS.test(agent.modelSpec(), agent.policyNetwork(),
                                  params, 50, 500)
            score = testScores["score"]
            agent.setScore(testScores)
            logger.startBlock()
            logger.log("Score is...")
            logger.logDict(testScores)
            logger.startBlock()
            trainingSpec.params = agent.params()
            logger.save(trainingSpec, ".spec")
            logger.save(agent, ".checkpoint")
            return score
        except Exception as ex:
            # if it fails it's usually due to NaNs in the net
            print("FAILED", ex)
            print("training spec is...")
            print(trainingSpec.toJSON())
            return 0.0

    # run a trials using our objective function
    study.optimize(objective, n_trials=args.nTrials)


################################################################################
async def runMinion(i: int, args: argparse.Namespace) -> None:
    """Run a subprocess """
    proc = await asyncio.create_subprocess_exec("./paramSearch.py", "-m", "-l",
                                                args.logdir,
                                                args.space, args.storage,
                                                str(args.nTrials))

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        if stdout:
            print(f'[stdout]\n{stdout.decode()}')
        if stderr:
            print(f'[stderr]\n{stderr.decode()}')


async def runMinions(args: argparse.Namespace) -> None:
    # run all our subprocess minions
    minions = []
    for i in range(args.nThreads):
        minions.append(runMinion(i, args))
    await asyncio.gather(*minions)


################################################################################
def main():
    parser = argparse.ArgumentParser(
        prog="paramSearch",
        description=
        "Search the hyper parameter space using the optuna algorithm.")
    parser.add_argument("space",
                        type=str,
                        default="a",
                        help="The definitions of our hyper param space")
    parser.add_argument(
        "storage",
        type=str,
        default="b",
        help=
        """The sqlite file to hold the optuna trials (or mysql URL). It will be created if it doesn't exist."""
    )
    parser.add_argument("nTrials",
                        type=int,
                        default=20,
                        help="The number of trials to run per thread")
    parser.add_argument(
        "-n",
        "--nThreads",
        type=int,
        default=1,
        help="The number of threads to use, only one will be used for A2C.")
    parser.add_argument('-m',
                        '--minion',
                        action='store_true',
                        help="Run a minion, !Don't use this directly!.")
    parser.add_argument(
        '-s',
        '--sampler',
        type=str,
        default="TPE",
        help="How to sample the hyper parameter space [random|TPE].")
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default="logs",
        help=
        "The directory to write logs and checkpoints into (it will make a sub directly with the name of the training space"
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help=
        "Reset the optuna storage, destroying all previous trial data for the training space."
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.minion:
        # don't have torch multithread as we will be running trials in parallel
        # and we will get higher throughput if we single thread torch, but run N trials
        # in parallel
        torch.set_num_threads(1)
        minion(args)
    else:
        with open(args.space, 'r') as file:
            trainingSpace = GTS.TrainingSpace.fromJSON(file.read())

        nThreads = args.nThreads
        if trainingSpace.technique.lower() == "a2c" and args.nThreads > 1:
            print(
                f"Can only run 1 trial in parallel for A2C as it multithreads."
            )
            nThreads = 1

        logdir = f"{args.logdir}/{trainingSpace.name}"

        print(f"Running {nThreads * args.nTrials} trials")
        print(f"Logging to directory {logdir}")
        # attempt to load the study
        study: optuna.study.Study | None = None

        # what is our storage
        storage = resolveStorage(args.storage)

        if args.reset:
            try:
                optuna.delete_study(study_name=trainingSpace.name,
                                    storage=storage)
            except:
                pass
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
        else:
            try:
                study = optuna.load_study(study_name=trainingSpace.name,
                                          storage=storage)
            except:
                pass

        # make study?
        if not study:
            sampler: optuna.samplers.BaseSampler | None = None
            if args.sampler.lower() == "random":
                sampler = optuna.samplers.RandomSampler()
            study = optuna.create_study(
                study_name=trainingSpace.name,
                storage=storage,
                direction=optuna.study.StudyDirection.MAXIMIZE,
                sampler=sampler)

        # make a log dir?
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            shutil.copy(args.space, logdir)

        print(f"Dispatching over {nThreads} thread.")

        # dispatch minion processes if nThreads > 1
        if nThreads > 1:
            asyncio.run(runMinions(args))
        else:
            minion(args)


if __name__ == "__main__":
    main()
