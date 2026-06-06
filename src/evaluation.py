from collections import Counter, OrderedDict
import numpy as np
from prettytable import PrettyTable
from prettytable import MARKDOWN

class Evaluation:
    def __init__(self):
        pass

    def get_rank_distribution(self, allocations: list, agents):
        # count agents per rank for a single run of matching
        # measure of rank depends on the rank reported in truthful or strategic preferences
        rank_counters = []
        agent_ranks = []
        for idx, alloc in enumerate(allocations):
            agent_rank = {a: None for a in alloc}
            current_agent = agents[idx]
            rank_counter = Counter()
            for agent in alloc:
                item, rank = alloc[agent]
                if rank == -1:
                    true_rank = -1
                elif item not in current_agent[agent]:
                    true_rank = -2
                else:
                    true_rank = current_agent[agent].index(item)
                rank_counter.update([true_rank])
                agent_rank[agent] = true_rank
            rank_counters.append(dict(rank_counter))
            agent_ranks.append(agent_rank)
        return rank_counters, agent_ranks

    def get_average_rank_distribution(self, allocations: list, agents):
        # count average agents per rank for multiple runs of matching
        # measure of rank depends on the rank reported in truthful or strategic preferences
        rc_avg = OrderedDict({})
        rc_err = OrderedDict({})
        rank_counters, _ = self.get_rank_distribution(allocations, agents)
        unique_keys = set()
        for rc in rank_counters:
            unique_keys.update(list(rc.keys()))
        unique_keys = sorted(list(unique_keys))
        max_rank = max(unique_keys)
        unique_keys = sorted(list(range(-2, max_rank + 1, 1)))
        for key in unique_keys:
            rc_avg[key] = np.average([x[key] if key in x else 0 for x in rank_counters])
            rc_err[key] = np.std([x[key] if key in x else 0 for x in rank_counters])
        print(rc_err)
        return rc_avg, rc_err, max_rank

    def get_average_rank(self, allocations, agents):
        average_ranks = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            average_rank = np.average([r if r >= 0 else len(agents[run][a]) - 1 for a, r in ar.items()])
            average_ranks.append(average_rank)
        return np.average(average_ranks)

    def get_max_rank(self, allocations, agents):
        max_ranks = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            max_rank = np.max([r if r >= 0 else len(agents[run][a]) - 1 for a, r in ar.items()])
            max_ranks.append(max_rank)
        return np.average(max_ranks)

    def get_top_n(self, allocations, n, agents):
        top_n_counts = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            top_n_count = 0
            c = [r for a, r in ar.items() if r >= 0]
            for i in range(n):
                top_n_count += c.count(i)
            top_n_counts.append(top_n_count/len(ar))
        return np.average(top_n_counts)

    def get_rank_n_plus(self, allocations, n, agents):
        bottom_counts = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            bottom_count = 0
            c = [r for a, r in ar.items() if r >= 0]
            for i in range(n, max(c) + 1):
                bottom_count += c.count(i)
            bottom_counts.append(bottom_count/len(ar))
        return np.average(bottom_counts)

    def get_not_placed(self, allocations, agents):
        not_placed = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            c = [r for a, r in ar.items() if r == -1]
            not_placed.append(len(c)/len(ar))
        return np.average(not_placed)

    def get_placed_elsewhere(self, allocations, agents):
        placed_elsewhere = []
        _, agent_ranks = self.get_rank_distribution(allocations, agents)
        for run, ar in enumerate(agent_ranks):
            c = [r for a, r in ar.items() if r == -2]
            placed_elsewhere.append(len(c)/len(ar))
        return np.average(placed_elsewhere)

    def get_table_header(self):
        t = PrettyTable(['Market', 'Method',
                         'Top-1', 'Top-3', 'Top-5', 'Rank 6+', 'Not Placed',
                         'Placed Elsewhere', 'Max Rank', 'Avg Rank'])
        t.set_style(MARKDOWN)
        return t
    def add_logs(self, market, method, matches, agents):
        row = [market, method,
               f"{self.get_top_n(matches, 1, agents) * 100:.2f}",
               f"{self.get_top_n(matches, 3, agents) * 100:.2f}", f"{self.get_top_n(matches, 5, agents) * 100:.2f}",
               f"{self.get_rank_n_plus(matches, 5, agents) * 100:.2f}",
               f"{self.get_not_placed(matches, agents) * 100:.2f}",
               f"{self.get_placed_elsewhere(matches, agents) * 100:.2f}", f"{self.get_max_rank(matches, agents)}",
               f"{self.get_average_rank(matches, agents):.2f}"]
        return row
