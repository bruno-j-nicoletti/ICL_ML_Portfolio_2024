#! /usr/bin/env python
import argparse
import torch
import os
import GTS


################################################################################
def main():
    parser = argparse.ArgumentParser(
        prog="check", description="List the contents of a checkpoint file.")
    parser.add_argument("checkpoint",
                        nargs='+',
                        type=str,
                        help="The previously trained checkpoints.")

    args = parser.parse_args()

    for fileName in args.checkpoint:
        # for each file
        if os.path.exists(fileName):
            # if it exists, load it and dump what is in there
            values = torch.load(fileName)
            agent = GTS.loadAgentFromDict(values)
            network = GTS.loadStochasticNetFromDict(values["policy.net"])
            print(f"File: {fileName}")
            for k in ["model", "trainer"]:
                print(f"{k} : {values[k]}")
            if "score" in values:
                print(f"Score")
                for k, v in values["score"].items():
                    print(f"   {k} : {v}")
            GTS.printParams(agent.params())

            if len(args.checkpoint) > 1:
                print("----------------------------------------")


if __name__ == "__main__":
    main()
