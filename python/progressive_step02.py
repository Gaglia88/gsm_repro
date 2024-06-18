import pandas as pd
import numpy as np
import random
import pickle
import os
from progressive_utils import Utils

class PruningUtils(object):
    @staticmethod
    def get_all_neighbors(profile_id, block, separators):
        """
        Given a block and a profile ID returns all its neighbors
        :param profile_id: profile id
        :param block: profile in which its contained
        :param separators: id of the separators that identifies the different data sources
        :return: all neighbors of the profile
        """
        output = set()
        i = 0
        while i < len(separators) and profile_id > separators[i]:
            output.update(block[i])
            i += 1
        i += 1

        while i < len(separators):
            output.update(block[i])
            i += 1

        if profile_id <= separators[-1]:
            output.update(block[-1])

        return output

class PPS(object):
    def __init__(self, profile_index, block_index, w_index, k=20, separator_ids=None):
        if separator_ids is None:
            self.separator_ids = []
        else:
            self.separator_ids = separator_ids

        self.profile_index = profile_index
        self.block_index = block_index
        self.w_index = w_index
        self.k = k
        self.comparison_list = list()
        self.sorted_profile_list = list()
        self.checked_entities = set()

    def get_all_neighbors(self, profile_id):
        """
        Given a profile_id returns all its neighbors
        """

        neighbors = set()
        # For every block in which the profile is contained
        for block_id in self.profile_index[profile_id]:
            if block_id in self.block_index:
                block_profiles = self.block_index[block_id]

                if len(self.separator_ids) == 0:
                    profiles_ids = block_profiles[0]
                else:
                    profiles_ids = PruningUtils.get_all_neighbors(profile_id, block_profiles, self.separator_ids)

                # Computes all the neighborhood
                neighbors = neighbors.union(profiles_ids)

        return set(filter(lambda x: profile_id < x, neighbors))

    def initialize(self):
        """
        Initialize PPS
        """
        top_comparisons = set()
        # For every profile
        for profile_id in self.profile_index:
            # Gets the neighborhood
            neighbors = self.get_all_neighbors(profile_id)

            # Gets the top-1 neighbor and computes the duplication_likelihood
            top_comp = (-1, -1, -1)
            # Neighbors that have the same weight of the top-1
            top_n = list()

            duplication_likelihood = 0

            for n_id in neighbors:
                if (profile_id, n_id) in self.w_index:
                    weight = self.w_index[(profile_id, n_id)]
                    duplication_likelihood += weight
                    if top_comp[2] < weight:
                        top_comp = (profile_id, n_id, weight)
                        top_n = list([n_id])
                    elif top_comp[2] == weight:
                        top_n.append(n_id)

            # If there is a valid comparison, adds it to the queue
            if top_comp[2] > 0:
                # If there are more neighbors with the same top-1 weight
                # picks up one randomly
                if len(top_n) > 1:
                    top_comp = (profile_id, random.choice(top_n), top_comp[2])
                top_comparisons.add(top_comp)

            if len(neighbors) > 0:
                duplication_likelihood /= len(neighbors)
                self.sorted_profile_list.append((profile_id, duplication_likelihood))

        # Sorts the comparisons list by the probability of being a match (highest first)
        self.comparison_list = list(top_comparisons)
        self.comparison_list = sorted(self.comparison_list, key=lambda x: -x[2])
        # Sorts the profiles by their duplication_likelihood (highest first)
        self.sorted_profile_list = sorted(self.sorted_profile_list, key=lambda x: -x[1])

    def get_next_comparison(self):
        """
        Returns the next most promising comparison
        """
        # There are no more comparisons to emit
        if len(self.comparison_list) == 0:
            # Computes the next comparisons
            if len(self.sorted_profile_list) > 0:
                profile_id = self.sorted_profile_list.pop(0)[0]
                self.checked_entities.add(profile_id)
                neighbors = self.get_all_neighbors(profile_id)
                sorted_stack = set()

                for n_id in neighbors:
                    if n_id not in self.checked_entities:
                        if (profile_id, n_id) in self.w_index:
                            sorted_stack.add((profile_id, n_id, self.w_index[(profile_id, n_id)], random.random()))

                # Randomizes the selection of the pairs with the same weight
                self.comparison_list = list(sorted_stack)
                self.comparison_list = sorted(self.comparison_list, key=lambda x: (-x[2], x[3]))
                self.comparison_list = list(map(lambda x: tuple(list(x)[:-1]), self.comparison_list))

                if len(self.comparison_list) > self.k:
                    self.comparison_list = self.comparison_list[:self.k]

        # Emit a comparison, if there are to emit
        if len(self.comparison_list) > 0:
            return self.comparison_list.pop(0)
        # No comparisons to emit, but there are profiles to explore
        elif len(self.sorted_profile_list) > 0:
            return self.get_next_comparison()
        # Done
        else:
            return (-1, -1, -1)


def get_pps_results(features, metric, profile_index, block_index, separator_ids, new_gt, n=-1):
    # Generates an index that given a pair returns their score (number of shared blocks)
    w_index = {}
    for index, row in features.iterrows():
        w_index[(row['p1'], row['p2'])] = row[metric]

    # PPS with CBS
    pps = PPS(profile_index, block_index, w_index, separator_ids=separator_ids)
    pps.initialize()

    res = []
    c = pps.get_next_comparison()
    new_gt1 = new_gt.copy()

    int_n = n

    while c[0] >= 0 and (int_n > 0 or n < 0):
        p = (c[0], c[1])
        if p in new_gt1:
            res.append(1)
            new_gt1.remove(p)
        else:
            res.append(0)

        c = pps.get_next_comparison()

        if n > 0:
            int_n -= 1
    return res

def make_process(base_path, dataset, all_features=["cfibf", "raccb", "js", "rs", "aejs", "nrs", "wjs"], budget=50):
    print(dataset)
    f = open(f'{base_path}/{dataset}/profile_index_{dataset}.pickle', 'rb')
    profile_index = pickle.load(f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/block_index_{dataset}.pickle', 'rb')
    block_index = pickle.load(f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/separator_ids_{dataset}.pickle', 'rb')
    separator_ids = pickle.load(f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/new_gt_{dataset}.pickle', 'rb')
    new_gt = pickle.load(f)
    f.close()
    
    features_no_train = pd.read_parquet(f'{base_path}/{dataset}/features.parquet')
    weights = pd.read_parquet(f'{base_path}/{dataset}/weights.parquet')

    num_matches = len(new_gt) - int(budget / 2)

    n = num_matches * 20    
    outdir = f"{base_path}/comparisons/{dataset}/"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    
    for i in range(0, 5):
        print(f"  Step {i}")
        
        if not os.path.isfile(f'{outdir}comp_sup_mb_run_{i}.pickle'):
            pps_sup_mb = get_pps_results(weights, "p_match", profile_index, block_index, separator_ids, new_gt, n=n)
            
            f = open(f'{outdir}comp_sup_mb_run_{i}.pickle', 'wb')
            pickle.dump(pps_sup_mb, f)
            f.close()
        
        
        for f in all_features:
            if not os.path.isfile(f'{outdir}comp_{f}_run_{i}.pickle'):
                res = get_pps_results(features_no_train, f, profile_index, block_index, separator_ids, new_gt, n=n)
                f = open(f'{outdir}comp_{f}_run_{i}.pickle', 'wb')
                pickle.dump(res, f)
                f.close()
    
if __name__ == '__main__':
    base_path = "/home/app/progressive/files"
    datasets = Utils.load_datasets()
    
    for d in datasets:
        make_process(base_path, d["name"])