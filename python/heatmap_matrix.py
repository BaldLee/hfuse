import matplotlib.pyplot as plt
import numpy as np
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file",
                        "-f",
                        required=True,
                        type=str,
                        help="Path of the json file to plot")
    args = parser.parse_args()
    filename = args.file
    with open(f"{filename}.json", "r") as fin:
        arr = json.load(fin)
    harvest = np.array(arr)
    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap="bwr")
    plt.colorbar(im)
    plt.savefig(f"{filename}.png")
