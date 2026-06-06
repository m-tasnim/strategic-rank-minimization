import random
import pandas as pd
from collections import OrderedDict
import numpy as np


class Strategy:
    def __init__(self, seed, agents, items, not_placed_id = None):
        self.agents = agents
        self.items = items
        self.seed = seed
        self.topn_modes = {
            "default": [18.5, 22.5, 23, 9, 14, 5, 2, 1.5],
            "uniform": [25, 25, 25, 25]
        }
        self.item_popularity = self.get_item_popularity(rank=0)
        random.seed(seed)
        self.not_placed_id = not_placed_id

    def get_item_popularity(self, rank: int = 0) -> dict:
        return {item: (sum([1 for pref in self.agents.values() if pref[rank] == item]) - self.items[item])
                for item in self.items.keys()}

    def get_strategic_agents(self, stype: str, fraction: float, topn: int, topn_mode: str = "default", cutoff: bool = True):
        random.seed(self.seed)
        strategic_agents = {a: self.agents[a] for a in self.agents}
        all_agents = sorted(list(self.agents.keys()))
        random.shuffle(all_agents)
        strategic_agent_ids = all_agents[:round(len(all_agents) * fraction)]
        if topn == "Mixed-n":

            n_perc = self.topn_modes[topn_mode]
            n_perc_sections = [round(sec / 100 * len(strategic_agent_ids)) for sec in n_perc]
            n_sagents = [list(k) for k in np.split(np.asarray(strategic_agent_ids, dtype=int),
                                                   np.cumsum(n_perc_sections))]
            #TODO: Rough hack because of Max's requested PR plots. Remove or make consistent.
            strategic_agent_ids = {}
            for index, s_agents in enumerate(n_sagents):
                strategic_agents = strategic_agents | self.__get_strategic_preferences(
                    sagents=s_agents,
                    stype=stype,
                    topn=index + 1,
                    cutoff=cutoff)
                strategic_agent_ids[index + 1] = s_agents
        else:
            strategic_agents = strategic_agents | self.__get_strategic_preferences(sagents=strategic_agent_ids,
                                                                                   stype=stype,
                                                                                   topn=topn,
                                                                                   cutoff=cutoff)
        return strategic_agents, strategic_agent_ids

    def __get_strategic_preferences(self, sagents: list, stype: str, topn: int, cutoff:bool=True) -> dict:
        random.seed(self.seed)
        strategic_prefs = {a: self.agents[a] for a in sagents}
        if stype == "Truthful":
            return strategic_prefs
        if stype == "FOH":
            for a in sagents:
                top_prefs = self.agents[a][:topn]
                s_items = [i for i in self.items.keys() if i not in top_prefs]
                if cutoff:
                    s_items = [i for i in s_items if self.item_popularity[i] > 0]
                else:
                    s_items = [i for i in s_items if self.item_popularity[i]]
                s_items.sort(key=lambda i: self.item_popularity[i], reverse=True)
                s_prefs = top_prefs + s_items
                if self.not_placed_id is not None:
                    if self.not_placed_id in s_prefs:
                        s_prefs.remove(self.not_placed_id)
                    strategic_prefs[a] = s_prefs + [self.not_placed_id]
                else:
                    strategic_prefs[a] = s_prefs
        if stype == "Reorder":
            for a in sagents:
                top_prefs = self.agents[a][:topn]
                s_items = [i for i in self.agents[a] if i not in top_prefs]
                if cutoff:
                    s_items = [i for i in s_items if self.item_popularity[i] > 0]
                else:
                    s_items = [i for i in s_items if self.item_popularity[i]]
                s_items.sort(key=lambda i: self.item_popularity[i], reverse=True)
                s_prefs = top_prefs + s_items
                if self.not_placed_id is not None:
                    if self.not_placed_id in s_prefs:
                        s_prefs.remove(self.not_placed_id)
                    strategic_prefs[a] = s_prefs + [self.not_placed_id]
                else:
                    strategic_prefs[a] = s_prefs
        return strategic_prefs


class SimulatedData:
    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)

    def __generate_preferences(self, agents, items, probabilities, not_placed_id):
        # assert sum(probabilities) == 1.0, "Probabilities do not sum to 1"
        for agent in agents.keys():
            prefs = list(np.random.choice(list(items.keys()), size=len(items), p=probabilities, replace=False))
            agents[agent] = prefs + [not_placed_id]
        return agents

    def generate_agents_and_items(self, n_agents, n_items, probabilities, scarcity=1):
        n_units = scarcity * n_agents
        items = {i: int(n_units / n_items) for i in range(n_items)}
        not_placed_id = n_items
        agents = {i: [] for i in range(n_agents)}
        agents = self.__generate_preferences(agents=agents, items=items, probabilities=probabilities, not_placed_id=not_placed_id)
        items[not_placed_id] = 10000
        return agents, items, not_placed_id


class SchoolChoiceData:
    """
    Data loading class for school choice data
    """

    def __init__(self, path):
        self.path = path
        self.student_df = None
        self.school_df = None

    def set_dataframes(self, student_df, school_df):
        self.student_df = student_df
        self.school_df = school_df

    def load_dataframes(self):
        """
        loads data from excel file
        :return: pandas dataframes containing school & student data
        """
        school_df = pd.read_excel(self.path, 'Klassen', engine='openpyxl')
        student_df = pd.read_excel(self.path, 'Resultaten', engine='openpyxl')
        student_df = student_df.set_index('Lotnummer')
        self.school_df = school_df
        self.student_df = student_df
        return self.school_df, self.student_df

    def load_preferences_capacities(self, advies=None):
        # TODO: Test this
        """
        load data from excel file and return preferences, capacities & advies
        :return: ordered dictionaries containing preferences and capacities
        key corresponds to student's lottery number
        Possible values of advies
        ['vwo',
         'vmbo - basis/kader',
         'vmbo - kader',
         'havo',
         'havo/vwo',
         'vmbo - basis',
         'vmbo - theoretisch',
         'vmbo - theoretisch/havo',
         'vmbo - basis + lwoo',
         'vmbo - basis/kader + lwoo',
         'vmbo - kader + lwoo',
         'vmbo - theoretisch + lwoo']
        """

        if advies:
            if advies == 'vmbo':
                vmbo_list = ['vmbo - basis/kader',
                             'vmbo - kader',
                             'vmbo - basis',
                             'vmbo - theoretisch',
                             'vmbo - theoretisch/havo',
                             'vmbo - basis + lwoo',
                             'vmbo - basis/kader + lwoo',
                             'vmbo - kader + lwoo',
                             'vmbo - theoretisch + lwoo']
                df_list = [self.student_df[self.student_df['Advies'] == vmbo] for vmbo in vmbo_list]
                student_df = pd.concat(df_list)
            elif advies == 'havo/vwo':
                havo_list = ['havo', 'havo/vwo']
                df_list = [self.student_df[self.student_df['Advies'] == havo] for havo in havo_list]
                student_df = pd.concat(df_list)
            else:
                student_df = self.student_df[self.student_df['Advies'] == advies]
        else:
            student_df = self.student_df

        # build preferences dict
        lotteries = sorted(student_df.index.tolist())
        preference_keys = [x for x in student_df.columns if 'Voorkeur' in x]
        student_dict = student_df.to_dict(orient='index')
        preferences = OrderedDict(
            {s: [student_dict[s][key] for key in preference_keys if not pd.isna(student_dict[s][key])] + ['Not Placed']
             for s in lotteries})

        if advies:
            unique_schools = pd.unique(student_df[preference_keys].values.ravel()).tolist()
            school_df = self.school_df[self.school_df['Key'].isin(unique_schools)]
        else:
            school_df = self.school_df

        # build capacities dict
        capacities = OrderedDict(zip(school_df.Key, school_df['Maximale capaciteit definitieve matching']))
        capacities['Not Placed'] = 10000

        schools = sorted(list(capacities.keys()))
        school_map = dict(zip(schools, list(range(len(schools)))))
        student_map = dict(zip(sorted(list(preferences.keys())), list(range(len(preferences)))))
        reverse_student_map = dict(zip(list(range(len(preferences))), sorted(list(preferences.keys()))))
        reverse_school_map = dict(zip(list(range(len(schools))), schools))

        coded_capacities = OrderedDict({school_map[s]: capacities[s] for s in schools})
        coded_prefs = OrderedDict({student_map[s]: [school_map[p] for p in v] for s, v in preferences.items()})

        return coded_prefs, coded_capacities, reverse_student_map, reverse_school_map
