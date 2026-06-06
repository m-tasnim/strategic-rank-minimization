from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import islice

class Matching:
    def __init__(self, agents: dict, items: dict, not_placed_id: int = None):
        self.agents = agents
        self.items = items
        self.not_placed_id = not_placed_id

    def random_priority(self) -> dict:
        """
        performs matching using random priority mechanism
        :return: dictionary containing agent id: (item id, rank of preferred item)
        """
        priorities = sorted(list(self.agents.keys()))
        tracker = Counter()
        allocation = {a: (-1, -1) for a in self.agents}
        for agent in priorities:
            for rank, item in enumerate(self.agents[agent]):
                if tracker[item] < self.items[item]:
                    if item == self.not_placed_id:
                        allocation[agent] = (item, -1)
                    else:
                        allocation[agent] = (item, rank)
                    tracker.update([item])
                    break
        return allocation

    # def probabilistic_serial(self) -> dict:
    #     """input is a list of preference lists"""
    #     agent_ids = sorted(list(self.agents.keys()))  # agents
    #     item_units = list(range(sum(self.items.values())))  # unique units of items
    #     it = iter(item_units)
    #     sliced = [list(islice(it, 0, i)) for i in self.items.values()]
    #     item_map = {i: sliced[i] for i in self.items}
    #     supply = {o: Fraction(1, 1) for o in item_units}
    #     fractional_allocation = {(i, o): Fraction(0, 1) for i in agent_ids for o in item_units}
    #     while any(supply.values()):
    #         # in each iteration, at least one remaining item is fully depleted
    #         eating = {}
    #         eaters = {o: 0 for o in item_units}  # number of agents eating each item
    #         for i in agent_ids:
    #             o_ = next((k for o in self.agents[i] for k in item_map[o] if supply[k]))
    #             eating[i] = o_
    #             eaters[o_] += 1
    #         # how much time until the first remaining item is depleted
    #         time = min(supply[o] / eaters[o] for o in item_units if supply[o] and eaters[o])
    #         for i in agent_ids:
    #             fractional_allocation[i, eating[i]] += time
    #             supply[eating[i]] -= time
    #     m = np.ones((len(agent_ids), len(item_units)))
    #     for i, o in fractional_allocation.keys():
    #         m[i, o] = fractional_allocation[(i, o)]
    #     decomposed = birkhoff_von_neumann_decomposition(m)
    #     choose = random.randint(0, len(decomposed))
    #     chosen_indices = decomposed[choose][1]
    #     allocation = {a: (-1, -1) for a in self.agents}
    #     for i, row in enumerate(chosen_indices):
    #         for j, column in enumerate(row):
    #             if chosen_indices[i, j] == 1:
    #                 allocated_item = [k for k in item_map.keys() if j in item_map[k]][0]
    #                 allocated_rank = self.agents[i].index(allocated_item)
    #                 allocation[i] = (allocated_item, allocated_rank)
    #     return allocation

    def hungarian_algorithm(self) -> dict:
        allocation = {a: (-1, -1) for a in self.agents}
        item_units = list(range(sum(self.items.values())))  # unique units of items
        it = iter(item_units)
        sliced = [list(islice(it, 0, i)) for i in self.items.values()]
        item_map = {i: sliced[i] for i in self.items}
        m = np.full(shape=(len(self.agents), len(item_units)), fill_value=99999, dtype=float)
        for a in self.agents:
            for i in self.agents[a]:
                if len(item_map[i]) > 0:
                    m[a][item_map[i][0]:item_map[i][-1]] = self.cost(a, i)
        row_ind, col_ind = linear_sum_assignment(m)
        for i, agent in enumerate(row_ind):
            unit = col_ind[i]
            item = [k for k in item_map.keys() if unit in item_map[k]][0]
            if item == self.not_placed_id:
                rank = -1
            else:
                try:
                    rank = self.agents[agent].index(item)
                except ValueError:
                    # print("ValueError: item not found in agents' preference list. Assigned rank: -2")
                    # print(item, self.agents[agent])
                    rank = -2
            allocation[agent] = (item, rank)
        return allocation

    def cost(self, agent, item, kind='linear'):
        if kind == 'linear':
            if item in self.agents[agent]:
                return self.agents[agent].index(item) + 1
            else:
                return 99999
    #%%
