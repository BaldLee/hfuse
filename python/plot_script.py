import matplotlib.pyplot as plt
import numpy as np
import json
from argparse import ArgumentParser
import math


def parser_addarg(parser: ArgumentParser):
    parser.add_argument("--file",
                        "-f",
                        required=True,
                        type=str,
                        help="Path of the json file to plot")


def main():

    def seqLenIndex(num) -> str:
        return "2^" + str(int(math.log2(num)))

    parser = ArgumentParser()
    parser_addarg(parser)
    args = parser.parse_args()
    json_path = args.file
    with open(json_path, "r") as jfin:
        jobj = json.load(jfin)
    seqLens = [1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25]
    heights = [256, 512, 1024, 2048, 4096, 8192, 16384]
    y1 = {}
    y2 = {}
    for i in seqLens:
        y1[seqLenIndex(i)] = []
        y2[seqLenIndex(i)] = []

    for i in jobj:
        y1[seqLenIndex(i["k2_total_elements"])].append(i["bncs_and_hist_res"])
        y2[seqLenIndex(i["k2_total_elements"])].append(i["hfuse_res"])

    fig = plt.figure(num=1)
    fig.set_size_inches(16, 10)
    ax = fig.add_subplot(231)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^20"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^20"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 7.8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^20")

    ax = fig.add_subplot(232)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^21"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^21"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 7.8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^21")

    ax = fig.add_subplot(233)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^22"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^22"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 7.8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^22")

    ax = fig.add_subplot(234)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^23"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^23"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 7.8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^23")

    ax = fig.add_subplot(235)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^24"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^24"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 7.8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^24")

    ax = fig.add_subplot(236)
    x = np.arange(len(heights))
    ax.plot(x, y1["2^25"], label="bncs&hist", linewidth=1, linestyle="solid")
    ax.plot(x, y2["2^25"], label="hfuse", linewidth=1, linestyle="solid")
    ax.set_ylim(0, 8)
    ax.set_xticks(np.linspace(0, 6, 7))
    ax.set_xticklabels(heights)
    ax.grid()
    ax.legend()
    ax.set_title("2^25")

    plt.savefig("out.png", dpi=300)


if __name__ == "__main__":
    main()
