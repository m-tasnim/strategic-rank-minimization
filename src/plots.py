from matplotlib import pyplot as plt
from src.evaluation import Evaluation
import numpy as np

class Plots:
    #TODO: bug: in plotter the error does not refresh between the loop
    def __init__(self):
        self.evaluation = Evaluation()

    def cumulative_plot(self, allocations, labels):
        f, ax = plt.subplots()
        rank_counters, _ = self.evaluation.get_rank_distribution(allocations=allocations)
        cumulative_sums = []
        for i, rc in enumerate(rank_counters):
            cs = np.cumsum(list(rc.values()))
            cs = cs / cs[-1]
            cumulative_sums.append(cs)
            ax.plot([r + 1 for r in range(len(cs))], cs, label=labels[i])
        return ax

    def average_cumulative_plot(self, allocations, labels=None, agents=None, ax=None, alpha=0.2, dynamic_ylim=None, cstyle=None):
        if not ax:
            _, ax = plt.subplots()
        rcs_avg = []
        rcs_err = []
        max_ranks = []

        for idx, alloc in enumerate(allocations):
            if agents:
                rc_avg, rc_err, max_rank = self.evaluation.get_average_rank_distribution(allocations=alloc, agents=agents[idx])
            else:
                rc_avg, rc_err, max_rank = self.evaluation.get_average_rank_distribution(allocations=alloc)
            rcs_avg.append(rc_avg)
            rcs_err.append(rc_err)
            max_ranks.append(max_rank)
        global_max_rank = max(max_ranks)
        if global_max_rank < 10:
            global_max_rank = 10
        ylim = []
        lines = []
        for i, rc in enumerate(rcs_avg):
            n_agents = sum(rc.values())
            ranks = [rc[rank] if rank in rc else 0 for rank in range(global_max_rank + 1)]
            errors = [rcs_err[i][rank] if rank in rcs_err[i] else 0 for rank in range(global_max_rank + 1)]
            cs = np.cumsum(ranks)
            cs_err = np.cumsum(errors)
            cs_err = cs_err / n_agents
            cs = cs / n_agents
            xticks = [r + 1 for r in range(len(cs))]
            ylim.append(min(cs))
            if cstyle and labels[i] in cstyle:
                f,  = ax.plot([r + 1 for r in range(len(cs))], cs, **cstyle[labels[i]])
            else:
                f,  = ax.plot([r + 1 for r in range(len(cs))], cs, label=labels[i], marker='.')
            lines.append(f)
            err_pos = cs + cs_err
            err_neg = cs - cs_err
            err_pos = [e if e < 1.0 else 1.0 for e in err_pos]
            err_neg = [e if e > 0.0 else 0.0 for e in err_neg]
            ax.fill_between([r + 1 for r in range(len(cs))], err_neg, err_pos, alpha=alpha)
            # xtick_step = 1 if max(xticks) < 15 else 3
            ax.set_xticks(range(min(xticks), max(xticks)+1, 2))
        if dynamic_ylim:
            min_y = min([k for k in ylim])
            ax.set_ylim([min_y, 1.005])
        return ax, lines













