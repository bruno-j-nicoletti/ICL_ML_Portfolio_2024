#! /usr/bin/env python
import argparse
import torch
import GTS


################################################################################
def main():
    parser = argparse.ArgumentParser(
        prog="play",
        description=
        "Play a trained a neural network checkpoint animating a gymnasium model."
    )
    parser.add_argument("checkpoint",
                        type=str,
                        help="The file holding the neural network checkpoint.")
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="The name of a video file to save to, default is to not save.")
    parser.add_argument("-r",
                        "--randomseed",
                        type=int,
                        default=0,
                        help="The random number seed to use.")
    parser.add_argument(
        "-k",
        "--keepplaying",
        action='store_true',
        help=
        "Keep playing after the simulation reports it died due to tripping over."
    )
    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=2,
        help=
        "If playing live, how many times to loop the simulation. Use 0 to loop forever."
    )
    parser.add_argument(
        "-n",
        "--nFrames",
        type=int,
        default=500,
        help="Number of frames to run in a loop or record to video.")

    args = parser.parse_args()

    # load it
    values = torch.load(args.checkpoint)
    model = values["model"]
    modelSpec = GTS.fetchPhysicalModelSpec(model)
    network = GTS.loadStochasticNetFromDict(values["policy.net"])

    # print info
    print(f"Playing {values['model']} trained with {values['trainer']}")
    if "score" in values:
        print("Score is", values["score"])
    GTS.printParams(values.get("params", {}))

    # play it
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
