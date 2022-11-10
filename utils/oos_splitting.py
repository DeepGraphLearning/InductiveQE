# taken from https://github.com/BorealisAI/OOS-KGE/blob/main/src/preprocess/dataset_prep.py
#
# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import os
import random

import numpy as np
import torch

"""
Code for dataset preparation. 
"""


class DatasetPreprocess:
    def __init__(self, dataset_triples, num_ent, num_rel, smpl_ratio=0.2, spl_ratio=0.5, inf_edges=0.15):
        #self.data_path = "datasets/" + dataset_name + "/"
        self.num_ent = num_ent
        self.num_rel = num_rel
        #self.all_triples = self.read_all()
        self.all_triples = np.array(dataset_triples, dtype=np.long)
        self.smpl_ratio = smpl_ratio
        self.spl_ratio = spl_ratio
        self.old_ent = []
        self.new_ent = []
        self.test_triples = []
        self.inference_edges_ratio = inf_edges

    def read_all(self):
        all_lines = []
        for spl in ["train", "valid", "test"]:
            file_path = self.data_path + spl + ".txt"
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    all_lines.append(line)
        triples = np.zeros((len(all_lines), 3), dtype=int)
        for i, line in enumerate(all_lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")))
        return triples

    def make_dataset(self):
        self.single_triple_ent()
        self.split_entities()
        self.separate_triples()
        self.find_dangling_ent()
        self.find_dangling_rel()
        self.balance_splits()

        self.normalize_ids()
        print("Done, access self.old_ent, self.old_triples for the transductive graph")
        print("Access self.new_val_ent and self.new_val_triples for validation graph")
        print("Access self.new_test_ent and self.new_test_triples for test graph")
        #self.explore_split_dataset()
        #self.constraint_check()
        #self.save_dataset()

    def single_triple_ent(self):
        """
        find those entities that are participated in only one triple
        add them to self.old_ent
        """
        #ent_triple_count = np.zeros(self.num_ent, dtype=int)
        ent_triple_count = np.zeros(self.num_ent, dtype=int)
        np.add.at(ent_triple_count, self.all_triples[:, 0], np.ones(len(self.all_triples), dtype=int))
        np.add.at(ent_triple_count, self.all_triples[:, 2], np.ones(len(self.all_triples), dtype=int))
        # for i in range(self.num_ent):
        #     ent_triple_count[i] = np.sum(self.all_triples[:, 0] == i) + np.sum(
        #         self.all_triples[:, 2] == i
        #     )
        # assert np.allclose(ent_triple_count, ent_triple_count2)
        single_triple_ent = np.where(ent_triple_count == 1)[0]
        self.old_ent.extend(list(single_triple_ent))

    def split_entities(self):
        all_ent = set(range(self.num_ent))
        all_ent = all_ent - set(self.old_ent)
        self.new_ent = random.sample(list(all_ent), int(len(all_ent) * self.smpl_ratio))
        self.old_ent.extend(list(all_ent - set(self.new_ent)))

        # sampling entities from those new nodes which will be in the test and validation sets
        self.new_test_ent = random.sample(self.new_ent, int(len(self.new_ent) * self.spl_ratio))
        self.new_val_ent = list(set(self.new_ent) - set(self.new_test_ent))

    def get_unique_entities(self, triples):
        h_ids = np.unique(triples[:, 0])
        t_ids = np.unique(triples[:, 2])
        triple_ids = np.union1d(h_ids, t_ids)
        return triple_ids

    def save_dataset(self):
        # save train
        new_dir = self.data_path + "processed/"
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        print("Saving old triples")
        old_triples_seq = [self.ids2triple(t) for t in self.old_triples]
        out_f = open(new_dir + "train.txt", "w")
        out_f.writelines(old_triples_seq)
        out_f.close()

        # split new triples to [val, test] and save
        print("Saving new triples")
        with open(new_dir + "valid.json", "w") as json_file:
            json.dump(self.valid_dict, json_file)
        with open(new_dir + "test.json", "w") as json_file:
            json.dump(self.test_dict, json_file)

    def constraint_check(self):
        old_ent = np.union1d(self.old_triples[:, 0], self.old_triples[:, 2])
        new_ent = set(self.test_dict.keys()).union(set(self.valid_dict.keys()))
        all_ent = len(old_ent) + len(new_ent)
        removed_ent = self.num_ent - all_ent
        print("New entity ratio: ", len(new_ent) / self.num_ent())
        print(
            "Number of deleted entities: {}, ratio: {}".format(
                removed_ent, removed_ent / self.num_ent
            )
        )

        total_triples = len(self.old_triples) + len(self.test_triples)
        removed_triples = len(self.all_triples) - total_triples
        print(
            "Number of deleted triples: {}, ratio: {}".format(
                removed_triples, removed_triples / len(self.all_triples)
            )
        )

        print(
            "[Train] #entities: {}, #triples: {}".format(
                len(self.old_ent), len(self.old_triples)
            )
        )
        print(
            "[Valid/Test] #entities: {}, #triples: {}".format(
                len(new_ent), len(self.test_triples)
            )
        )

    def separate_triples(self):
        self.new_triples, new_ids = self.get_ent_triples(self.new_ent, self.all_triples, return_ids=True)

        self.new_test_triples, test_ids = self.get_ent_triples(self.new_test_ent, self.new_triples, return_ids=True)
        self.new_val_triples, val_ids = self.get_ent_triples(self.new_val_ent, self.new_triples, return_ids=True)

        # TODO separate val triples from test triples: test should only be connected to train, not val nodes
        mask = np.ones(len(self.all_triples), dtype=bool)
        mask[new_ids] = False
        self.old_triples = self.all_triples[mask]

    def find_dangling_ent(self):
        old_triples_ent = np.union1d(self.old_triples[:, 0], self.old_triples[:, 2])
        self.dang_ent = list(set(self.old_ent) - set(old_triples_ent))

    def find_dangling_rel(self):
        old_triples_rel = set(self.old_triples[:, 1])
        rel_ids = list(range(self.num_rel))
        self.dang_rel = list(set(rel_ids) - old_triples_rel)

    def balance_splits(self):

        # remove triples with dangle entities from validation
        _, ids = self.get_ent_triples(self.dang_ent, self.new_val_triples, return_ids=True)
        mask = np.ones(len(self.new_val_triples), dtype=bool)
        mask[ids] = False
        self.new_val_triples = self.new_val_triples[mask]

        # remove triple with dangle entities from test
        _, ids = self.get_ent_triples(self.dang_ent, self.new_test_triples, return_ids=True)
        mask = np.ones(len(self.new_test_triples), dtype=bool)
        mask[ids] = False
        self.new_test_triples = self.new_test_triples[mask]

        # remove triples from val that contain dangling relations
        ids = np.nonzero(np.in1d(self.new_val_triples[:, 1], self.dang_rel))
        mask = np.ones(len(self.new_val_triples), dtype=bool)
        mask[ids] = False
        self.new_val_triples = self.new_val_triples[mask]

        # remove triples from test that contain dangling relations
        ids = np.nonzero(np.in1d(self.new_test_triples[:, 1], self.dang_rel))
        mask = np.ones(len(self.new_test_triples), dtype=bool)
        mask[ids] = False
        self.new_test_triples = self.new_test_triples[mask]

        # remove triples from test that contain validation entities
        _, ids = self.get_ent_triples(self.new_val_ent, self.new_test_triples, return_ids=True)
        mask = np.ones(len(self.new_test_triples), dtype=bool)
        mask[ids] = False
        self.new_test_triples = self.new_test_triples[mask]

        # remove triples from validation that contain test entities
        _, ids = self.get_ent_triples(self.new_test_ent, self.new_val_triples, return_ids=True)
        mask = np.ones(len(self.new_val_triples), dtype=bool)
        mask[ids] = False
        self.new_val_triples = self.new_val_triples[mask]

        unique_train_relations = np.unique(self.old_triples[:, 1])
        # remove triples from test that contain relations unseen in train
        ids = np.nonzero(np.in1d(self.new_test_triples[:, 1], unique_train_relations))
        mask = np.zeros(len(self.new_test_triples), dtype=bool)
        mask[ids] = True
        self.new_test_triples = self.new_test_triples[mask]

        # remove triples from validation that contain relations unseen in train
        ids = np.nonzero(np.in1d(self.new_val_triples[:, 1], unique_train_relations))
        mask = np.zeros(len(self.new_val_triples), dtype=bool)
        mask[ids] = True
        self.new_val_triples = self.new_val_triples[mask]

        # split to inference graph and edges to predict
        self.new_val_triples = np.random.permutation(self.new_val_triples)
        self.val_inference = self.new_val_triples[:int(len(self.new_val_triples)*(1-self.inference_edges_ratio)), :]
        self.val_predict = self.new_val_triples[int(len(self.new_val_triples)*(1-self.inference_edges_ratio)):, :]


        self.new_test_triples = np.random.permutation(self.new_test_triples)
        self.test_inference = self.new_test_triples[:int(len(self.new_test_triples)*(1-self.inference_edges_ratio)), :]
        self.test_predict = self.new_test_triples[int(len(self.new_test_triples)*(1-self.inference_edges_ratio)):, :]


        # make sure val_predict and test_predict do not contain unseen entities

        # assert self.get_unique_entities(self.val_predict).issubset(
        #     self.get_unique_entities(np.concatenate([self.old_triples, self.val_inference], axis=0))
        # )
        # assert self.get_unique_entities(self.test_predict).issubset(
        #     self.get_unique_entities(np.concatenate([self.old_triples, self.test_inference], axis=0))
        # )

        unseen_nodes = set(self.get_unique_entities(self.val_predict)).difference(
            set(self.get_unique_entities(
                np.concatenate([self.old_triples, self.val_inference], axis=0)
            ))
        )
        _, ids = self.get_ent_triples(np.array(list(unseen_nodes), dtype=int), self.val_predict, return_ids=True)
        mask = np.zeros(len(self.val_predict), dtype=bool)
        mask[ids] = True
        self.val_to_add = self.val_predict[mask]
        self.val_predict = self.val_predict[~mask]
        self.val_inference = np.concatenate([self.val_inference, self.val_to_add], axis=0)

        # the same for test
        unseen_nodes = set(self.get_unique_entities(self.test_predict)).difference(
            set(self.get_unique_entities(
                np.concatenate([self.old_triples, self.test_inference], axis=0)
            ))
        )
        _, ids = self.get_ent_triples(np.array(list(unseen_nodes), dtype=int), self.test_predict, return_ids=True)

        mask = np.zeros(len(self.test_predict), dtype=bool)
        mask[ids] = True
        self.test_to_add = self.test_predict[mask]
        self.test_predict = self.test_predict[~mask]
        self.test_inference = np.concatenate([self.test_inference, self.test_to_add], axis=0)




    def normalize_ids(self):
        # triples in the splits do not have contiguous IDs after preprocessing
        # ensure the splits are in the range 0 - N

        print("Remapping to contiguous IDs")
        train_ents = np.unique(self.old_triples[:, [0, 2]]).tolist()
        train_rels = np.unique(self.old_triples[:, 1]).tolist()
        val_inference_ents = np.unique(self.val_inference[:, [0, 2]]).tolist()
        test_inference_ents = np.unique(self.test_inference[:, [0, 2]]).tolist()

        new_inference_ents = sorted(set(val_inference_ents).difference(set(train_ents)))
        new_test_ents = sorted(set(test_inference_ents).difference(set(train_ents)))

        entity_mapping = {k: i for i,k in enumerate(train_ents + new_inference_ents + new_test_ents)}
        rel_mapping = {k: i for i,k in enumerate(train_rels)}

        self.backup = [copy.deepcopy(x) for x in (self.old_triples, self.val_inference, self.val_predict, self.test_inference, self.test_predict)]

        self.old_triples = self.remap(self.old_triples, entity_mapping, rel_mapping)
        self.val_inference = self.remap(self.val_inference, entity_mapping, rel_mapping)
        self.val_predict = self.remap(self.val_predict, entity_mapping, rel_mapping)
        self.test_inference = self.remap(self.test_inference, entity_mapping, rel_mapping)
        self.test_predict = self.remap(self.test_predict, entity_mapping, rel_mapping)

        self.old_triples, self.val_inference, self.val_predict, self.test_inference, self.test_predict \
            = self.old_triples.tolist(), self.val_inference.tolist(), self.val_predict.tolist(), self.test_inference.tolist(), self.test_predict.tolist()
        self.new_val_triples = (self.val_inference, self.val_predict)
        self.new_test_triples = (self.test_inference, self.test_predict)

        self.global_e2id = entity_mapping
        self.global_r2id = rel_mapping
        self.ent_splits = (len(train_ents), len(new_inference_ents), len(new_test_ents))


    def remap(self, triples, ent_mapping, rel_mapping):

        result = np.zeros_like(triples)
        for i, row in enumerate(triples):
            result[i, 0] = ent_mapping[row[0]]
            result[i, 1] = rel_mapping[row[1]]
            result[i, 2] = ent_mapping[row[2]]

        return result


    def explore_split_dataset(self):
        new_ent_dict = {}
        for new_e in self.new_ent:
            ent_triples = self.get_ent_triples([new_e], self.new_triples)
            # remove those triples that contain dangle entity
            _, ids = self.get_ent_triples(self.dang_ent, ent_triples, return_ids=True)
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]
            # remove those triples that contain dangle relations
            ids = np.nonzero(np.in1d(ent_triples[:, 1], self.dang_rel))
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]
            # remove those triples that contain other new_ent
            other_new_ent = list(set(self.new_ent) - set([new_e]))
            _, ids = self.get_ent_triples(other_new_ent, ent_triples, return_ids=True)
            mask = np.ones(len(ent_triples), dtype=bool)
            mask[ids] = False
            ent_triples = ent_triples[mask]

            # remove reflexive triples
            # ref_ids = np.where(ent_triples[:, 0] == ent_triples[:, 2])
            # mask = np.ones(len(ent_triples), dtype=bool)
            # mask[ref_ids] = False
            # ent_triples = ent_triples[mask]

            if len(ent_triples) >= 2:
                new_ent_dict[new_e] = [
                    [t[0], t[1], t[2]]
                    for t in ent_triples
                ]
                self.test_triples.extend(ent_triples.tolist())

        new_keys = list(new_ent_dict.keys())
        valid_ent = random.sample(new_keys, int(len(new_keys) * self.spl_ratio))
        test_ent = list(set(new_keys) - set(valid_ent))
        self.valid_dict = {k: new_ent_dict[k] for k in valid_ent}
        self.test_dict = {k: new_ent_dict[k] for k in test_ent}

    def get_ent_triples(self, e_ids, triples, return_ids=False):
        h_ids = np.nonzero(np.in1d(triples[:, 0], e_ids))
        t_ids = np.nonzero(np.in1d(triples[:, 2], e_ids))
        triple_ids = np.union1d(h_ids, t_ids)
        if return_ids:
            return triples[triple_ids], triple_ids
        return triples[triple_ids]

    def smpl_new_ent(self):
        all_keys = self.ent2id.keys()
        new_keys = random.sample(all_keys, int(self.num_ent() * self.smpl_ratio))
        old_keys = set(all_keys) - set(new_keys)
        self.new_ent2id = {k: self.ent2id[k] for k in new_keys}
        self.old_ent2id = {k: self.ent2id[k] for k in old_keys}
        self.new_ent = list(self.new_ent2id.values())
        self.old_ent = list(self.old_ent2id.values())


    def triple2ids(self, triple):
        return [
            self.get_ent_id(triple[0]),
            self.get_rel_id(triple[1]),
            self.get_ent_id(triple[2]),
        ]

    def ids2triple(self, ids):
        return "{0}\t{1}\t{2}\n".format(
            self.get_ent_str(ids[0]), self.get_rel_str(ids[1]), self.get_ent_str(ids[2])
        )

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def get_ent_str(self, e_id):
        for key, value in self.ent2id.items():
            if value == e_id:
                return key

    def get_rel_str(self, r_id):
        for key, value in self.rel2id.items():
            if value == r_id:
                return key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset", default="YAGO3-10", type=str, help="wordnet dataset"
    )
    parser.add_argument(
        "-smpl_ratio", default=0.2, type=float, help="new entities ratio"
    )
    parser.add_argument(
        "-spl_ratio", default=0.5, type=float, help="new dataset split ratio"
    )
    args = parser.parse_args()

    print("sample ratio: ", args.smpl_ratio)
    dataset_prep = DatasetPreprocess(
        args.dataset, smpl_ratio=args.smpl_ratio, spl_ratio=args.spl_ratio
    )
    print("saving datasets...", args.dataset)
    dataset_prep.make_dataset()

    print("done!")
