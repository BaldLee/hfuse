import matplotlib.pyplot as plt
import numpy as np
import re


class TunningConfig:

    def __init__(self) -> None:
        self.type: str = "unknow"
        self.k1TotalElements: int = 0
        self.k2TotalElements: int = 0
        self.grids: dict[int, dict] = {}


def loadLog(file_path) -> list[TunningConfig]:
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    configs = []

    for line in lines:
        r0 = r'(\d+)\s+(\d+)'
        r1 = r'gridDim\[(\d+)\] memory: ([\d.]+) computation: ([\d.]+)'
        r2 = r'gridDim\[(\d+)\] memory: ([\d.]+),([\d.]+) computation: ([\d.]+),([\d.]+)'
        res0 = re.match(r0, line)
        res1 = re.match(r1, line)
        res2 = re.match(r2, line)
        if res0:
            print(f"{res0[1]} {res0[2]}")
            config = TunningConfig()
            configs.append(config)
            config.k1TotalElements = int(res0[1])
            config.k2TotalElements = int(res0[2])
        if res1:
            print(f"{res1[1]} {res1[2]} {res1[3]}")
            config.type = "hfuse"
            grid = {}
            grid["mem"] = float(res1[2]) / 100
            grid["com"] = float(res1[3]) / 100
            config.grids[int(res1[1])] = grid
        if res2:
            print(f"{res2[1]} {res2[2]} {res2[3]} {res2[4]} {res2[5]}")
            config.type = "stream"
            grid = {}
            grid["k1"] = {}
            grid["k2"] = {}
            grid["k1"]["mem"] = float(res2[2]) / 100
            grid["k2"]["mem"] = float(res2[3]) / 100
            grid["k1"]["com"] = float(res2[4]) / 100
            grid["k2"]["com"] = float(res2[5]) / 100
            config.grids[int(res2[1])] = grid

    return configs


def plot_hfuse(config: TunningConfig):
    title = f"{config.k1TotalElements}_{config.k2TotalElements}"
    gridx = list(config.grids.keys())
    ymem, ycom = [], []
    for i in gridx:
        ymem.append(config.grids[i]["mem"])
        ycom.append(config.grids[i]["com"])

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    x = np.arange(len(gridx))
    color1 = "blue"
    color2 = "gold"
    lns1 = ax.plot(x, ymem, label="mem", linestyle="solid", color=color1)
    lns2 = ax.plot(x, ycom, label="com", linestyle="dashed", color=color2)

    ax.set_xticks(np.linspace(0, len(gridx) - 1, len(gridx)))
    ax.set_xticklabels(gridx)
    ax.set_ylim(0, 0.55)

    ax.legend()
    plt.title(title)
    plt.savefig(f"{title}.png")


def plot_stream(config: TunningConfig):
    title = f"{config.k1TotalElements}_{config.k2TotalElements}"
    gridx = list(config.grids.keys())
    y1mem, y1com, y2mem, y2com = [], [], [], []
    for i in gridx:
        y1mem.append(config.grids[i]["k1"]["mem"])
        y1com.append(config.grids[i]["k1"]["com"])
        y2mem.append(config.grids[i]["k2"]["mem"])
        y2com.append(config.grids[i]["k2"]["com"])

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin = ax.twinx()

    x = np.arange(len(gridx))
    color1 = "blue"
    color2 = "gold"
    lns1 = ax.plot(x, y1mem, label="y1mem", linestyle="solid", color=color1)
    lns2 = ax.plot(x, y2com, label="y1com", linestyle="dashed", color=color2)
    lns3 = twin.plot(x, y2com, label="y2mem", linestyle="solid", color=color2)
    lns4 = twin.plot(x, y2com, label="y2com", linestyle="dashed", color=color1)

    ax.set_xticks(np.linspace(0, len(gridx) - 1, len(gridx)))
    ax.set_xticklabels(gridx)
    ax.set_ylim(0, 0.55)
    twin.set_ylim(0, 0.15)

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    plt.title(title)
    plt.savefig(f"{title}.png")


def main():
    configs = loadLog('tunning_res/res.log')
    for config in configs:
        if config.type == "hfuse":
            plot_hfuse(config)
        if config.type == "stream":
            plot_stream(config)


if __name__ == "__main__":
    main()
