#! /usr/bin/env python
import argparse
import torch
import GTS


################################################################################
def main():
    parser = argparse.ArgumentParser(
        prog="play",
        description=
        "Play a trained a neural network animating a physical model.")
    parser.add_argument(
        "checkpoint",
        type=str,
        help=
        "The saved neural network trained to animate the model. Previously saved by trainer."
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="The name of the video file to save to, default is to not save.")
    parser.add_argument("-r",
                        "--randomseed",
                        type=int,
                        default=0,
                        help="The random number seed to use.")
    parser.add_argument(
        "-k",
        "--keepplaying",
        action='store_true',
        help="Keep playing after the simulation reports it died.")
    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=2,
        help=
        "If playing live, how many times to loop the simulation. Use 0 to loop forever."
    )
    parser.add_argument("-n",
                        "--nFrames",
                        type=int,
                        default=500,
                        help="Number of frames to run in a loop.")

    args = parser.parse_args()

    values = torch.load(args.checkpoint)
    model = values["model"]
    modelSpec = GTS.fetchPhysicalModelSpec(model)
    network = GTS.loadStochasticNetFromDict(values["policy.net"])

    print(f"Playing {values['model']} trained with {values['trainer']}")
    if "score" in values:
        print("Score is", values["score"])
    GTS.printParams(values.get("params", {}))

    GTS.play(modelSpec,
             network,
             seed=args.randomseed,
             stochastic=False,
             videoFileName=args.video,
             nFrames=args.nFrames,
             loops=args.loops,
             breakOnTermination=not args.keepplaying)


if __name__ == "__main__":
    main()
