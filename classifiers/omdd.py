#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Ordered Multi-valued Decision Diagrams (OMDDs)
#   Author: Xuanxiang Huang
#
################################################################################
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
import itertools
import csv
from queue import Queue
################################################################################


class OMDD(object):
    """
        OMDD classifiers.
    """

    def __init__(self, graph, root, nfeats, features, feat_domain,
                 target, tar_range, lvl2feat, feat2lvl, verb=0):
        self.graph = graph                  # MDD (use multi-edge)
        self.root = root                    # root node
        self.nf = nfeats                    # number of features
        self.features = features            # feature names
        self.feat_domain = feat_domain      # feature domain
        self.fv_probs = dict()              # feature-value probabilities
        self.target = target                # target name
        self.tar_range = tar_range          # range of target
        self.lvl2feat = lvl2feat            # level to feature (start from 0, top is level 0, increasing down)
        self.feat2lvl = feat2lvl            # feature to level
        self.verbose = verb

    @classmethod
    def from_file(cls, filename):
        """
            Load OMDD file.

            :param filename: file in .mdd format.
            :return: OMDD model.
        """

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        # filtering out comment lines (those that start with '#')
        lines = list(filter(lambda l: (not (l.startswith('#') or l.strip() == '')), lines))

        # auxiliary
        attr2var = dict()
        var2attr = dict()
        cla2tm = dict()
        attributes = []         # consistent with dataset, last attribute is the target
        attr_domain = dict()    # domain of each attribute
        attr2lvl = dict()       # attribute to level, the target is the last level (level 1)
        lvl2attr = dict()       # level to attribute
        # all nodes and edges
        t_nds = []
        nt_nds = []
        edges = []
        all_ts = []
        all_nts = []

        index = 0
        assert (lines[index].strip().startswith('// attributes domain:'))
        index += 1

        ########## get attribute name, attribute domain ##########
        while not lines[index].strip().startswith('(Forest Addr:'):
            assert lines[index].strip().startswith('// ')
            attr_line = lines[index].strip().lstrip('// ')
            domain_info = attr_line.split(';')
            assert len(domain_info) == 4, "incorrect format"
            attr = domain_info[0].strip()
            attr_dom = [int(v) for v in domain_info[1].strip().split(': ')[1].lstrip('[').rstrip(']').split(',')]
            dom_size = int(domain_info[2].strip().split(': ')[1])
            assert len(attr_dom) == dom_size
            var = int(domain_info[-1].split(':')[1].strip())
            attributes.append(attr)
            attr_domain.update({attr: attr_dom})
            attr2var.update({attr: var})
            var2attr.update({var: attr})
            index += 1
        attributes.reverse()

        assert lines[index].strip().startswith('(Forest Addr:')
        index += 1
        assert lines[index].strip().startswith('MTMDD rooted')
        index += 1
        ########## get attribute name, attribute domain ##########

        ########## get terminal node ##########
        tmp_idx = -1
        while not lines[tmp_idx].startswith('Level: 1 '):
            term_line = lines[tmp_idx].strip().split(": ")
            nd = int(term_line[1].rstrip(' down'))
            label = term_line[-1].strip().lstrip('(').rstrip(')').split(":")
            assert label[-1] == 'T'
            val = int(label[0])
            assert val in attr_domain[attributes[-1]]
            assert nd not in all_ts
            all_ts.append(nd)
            t_nds.append(tuple((nd, {'target': val})))
            cla2tm.update({val: nd})
            tmp_idx -= 1
        assert lines[tmp_idx].startswith('Level: 1 Var: 1')
        attr2lvl.update({var2attr[1]: 1})
        lvl2attr.update({1: var2attr[1]})
        ########## get terminal node ##########

        var_now = None
        lvl_now = None
        root = None
        while not lines[index].startswith('Level: 1 '):
            mdd_line = lines[index].strip()
            if mdd_line.startswith('Level'):
                l_v = mdd_line.split()
                lvl = int(l_v[1])
                var = int(l_v[-1])
                attr = var2attr[var]
                lvl2attr.update({lvl: attr})
                attr2lvl.update({attr: lvl})
                var_now = var
                lvl_now = lvl
            elif mdd_line.startswith('node:'):
                attr = var2attr[var_now]
                dom = attr_domain[attr]
                nd_line = mdd_line.split(": ")
                assert len(nd_line) == 3, "incorrect format"
                nd = int(nd_line[1].rstrip(' down'))
                assert nd not in all_nts
                all_nts.append(nd)
                nt_nds.append(tuple((nd, {'var': attr})))
                if lvl_now == len(attributes):
                    root = nd
                succ_line = nd_line[-1]
                if succ_line.startswith('['):
                    succs = succ_line.lstrip('[').rstrip(']').split('|')
                    assert len(succs) == len(dom)
                    assert 'T' not in succs
                    # only 'F' and node index (node index may refer to terminal node)
                    # if all children are 'F' or terminal nodes,
                    # then force all children are not the same terminal nodes.
                    missing_ts = set()
                    for t in all_ts:
                        if t not in succs:
                            missing_ts.add(t)
                    for val, chd in zip(dom, succs):
                        if chd == 'F':
                            if len(missing_ts):
                                real_chd = missing_ts.pop()
                                edges.append(tuple((nd, real_chd, val)))
                            else:
                                # F here means unknown, Randomly pick a value from class label
                                rand_class = random.choice(attr_domain[attributes[-1]])
                                real_chd = cla2tm[rand_class]
                                edges.append(tuple((nd, real_chd, val)))
                        else:
                            edges.append(tuple((nd, int(chd), val)))
                elif succ_line.startswith('('):
                    succs = succ_line.lstrip('(').rstrip(')').split(', ')
                    if len(succs) == len(dom):
                        for item in succs:
                            v_c = item.split(':')
                            val = int(v_c[0])
                            chd = int(v_c[1])
                            assert val in dom
                            edges.append(tuple((nd, chd, val)))
                    else:
                        missing_ts = set(all_ts)
                        missing_vals = set(dom)
                        for item in succs:
                            v_c = item.split(':')
                            val = int(v_c[0])
                            chd = int(v_c[1])
                            assert val in dom
                            missing_vals.remove(val)
                            if chd in missing_ts:
                                missing_ts.remove(chd)
                            edges.append(tuple((nd, chd, val)))
                        for val in missing_vals:
                            if len(missing_ts):
                                real_chd = missing_ts.pop()
                                edges.append(tuple((nd, real_chd, val)))
                            else:
                                # F here means unknown, Randomly pick a value from class label
                                rand_class = random.choice(attr_domain[attributes[-1]])
                                real_chd = cla2tm[rand_class]
                                edges.append(tuple((nd, real_chd, val)))
                else:
                    assert False, f"format: {succ_line} seems incorrect"

            index += 1

        ##### construct OMDD #####
        G = nx.MultiDiGraph()
        G.add_nodes_from(t_nds)
        G.add_nodes_from(nt_nds)
        G.add_edges_from(edges)
        ##### construct OMDD #####

        ########## check MDD: ##########
        # 1) for each node, in-degree > 0, except root node has 0 in-degree;
        # 2) for each node, out-degree > 0,  except terminal node has 0 out-degree.
        # 3) every node is reachable.
        # 4) for each non-terminal nodes, the number of outgoing edges == domain size.
        # 6) for each non-terminal nodes, all children nodes are not the same.
        assert len(G.nodes) == len(all_ts) + len(all_nts)
        for nd in G.nodes:
            if nd == root:
                assert G.in_degree(nd) == 0
            else:
                assert G.in_degree(nd) > 0
            if G.out_degree(nd) == 0:
                assert nd in all_ts
            else:
                assert nd in all_nts
        for nd in G.nodes:
            if G.out_degree(nd):
                all_chds = set(G.successors(nd))
                assert len(all_chds) > 1
                attr = G.nodes[nd]['var']
                attr_dom = attr_domain[attr]
                all_egs = set()
                for chd in G.successors(nd):
                    for val in attr_dom:
                        if tuple((nd, chd, val)) in G.edges:
                            all_egs.add(tuple((nd, chd, val)))
                assert len(all_egs) == len(attr_dom)
        ########## check MDD: ##########

        ########## features, feature domain, target, target range ##########
        feat_domain = dict()
        lvl2feat = dict()
        feat2lvl = dict()
        features = attributes[:]
        target = features.pop()
        for feat in features:
            assert feat in attr_domain
            feat_domain.update({feat: attr_domain[feat]})
        assert target in attr_domain
        tar_range = attr_domain[target]
        tar_range.sort()
        lvl2attr.pop(attr2lvl[target], None)
        attr2lvl.pop(target, None)
        order_feats = features[:]
        order_feats.sort(key=attr2lvl.get, reverse=True)
        for lvl, feat in enumerate(order_feats):
            lvl2feat.update({lvl: feat})
            feat2lvl.update({feat: lvl})
            assert attr2lvl[feat] >= 2
            assert lvl == len(features)-(attr2lvl[feat]-1)
        ########## features, feature domain, target, target range ##########

        return cls(G, root, len(features), features, feat_domain, target, tar_range, lvl2feat, feat2lvl)

    def set_fv_probs_uniform(self):
        """
            Set the feature-value probabilities to be uniformed.
        """
        for var in self.feat_domain:
            dom = self.feat_domain[var]
            prob = len(dom) * [1 / len(dom)]
            self.fv_probs.update({var: prob})

    def set_fv_probs(self, fv_probs):
        """
            Set the feature-value probabilities.
            :param fv_probs: a dictionary of feature-value probabilities.
        """
        self.fv_probs = fv_probs

    def gen_function(self, filename):
        """
            Generate the function represented by this OMDD.
            :return: a dictionary of function represented by this OMDD.
        """
        truth_table = []
        domains = []
        for feat in self.features:
            domains.append(self.feat_domain[feat])

        for inp in itertools.product(*domains):
            output = self.total_assignment(inp)
            truth_table.append(list(inp) + [output])

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.features + [self.target])
            writer.writerows(truth_table)

    def total_assignment(self, assignment):
        """
            Assign values to variables and reach a terminal node.
            :return: label (an integer) of terminal.
        """

        nd = self.root
        G = self.graph
        while G.out_degree(nd):
            for chd in G.successors(nd):
                f_id = self.features.index(G.nodes[nd]['var'])
                val = assignment[f_id]
                if tuple((nd, chd, val)) in G.edges:
                    nd = chd
                    break
            else:
                assert False, 'dead end branch'
        assert G.out_degree(nd) == 0
        return G.nodes[nd]['target']

    def predict(self, data_points):
        """
            Return a list of prediction given a list of data points.
            :param data_points: input data points
            :return: predictions of these data points.
        """
        data_pts = data_points
        if type(data_points) == pd.DataFrame:
            data_pts = data_points.to_numpy()
        predictions = []
        for dp in data_pts:
            predictions.append(self.total_assignment([int(e) for e in list(dp)]))
        return np.array(predictions)

    def accuracy(self, in_x, y_true):
        """
            Compare the output of bdd and desired prediction
            :param in_x: a list of total data points.
            :param y_true: desired prediction
            :return: accuracy in float.
        """

        y_pred = []
        for ins in in_x:
            assignment = [int(pt) for pt in ins]
            y_pred.append(self.total_assignment(assignment))
        acc = accuracy_score(y_true, y_pred)
        return acc

    def path_to_other_class(self, data_pt, tar, univ):
        """
            Check whether there is a path to A TARGET VALUE
            that differs from the target value of the given data point.

            :param data_pt: given data point.
            :param tar: target value of the given data point.
            :param univ: a list of features declared as universal.
            :return: true if there is a path to 0 else false.
        """

        G = self.graph
        # BFS (Breadth-first search)
        q = Queue()
        q.put(self.root)
        while not q.empty():
            nd = q.get()
            if not G.out_degree(nd):
                if G.nodes[nd]['target'] != tar:
                    return True
            else:
                f_id = self.features.index(G.nodes[nd]['var'])
                val = data_pt[f_id]
                if f_id in univ:
                    for chd in G.successors(nd):
                        q.put(chd)
                else:
                    for chd in G.successors(nd):
                        if tuple((nd, chd, val)) in G.edges:
                            q.put(chd)
                            break
                    else:
                        assert False, 'dead end branch'
        return False

    def dfs_postorder(self, root):
        """
            Iterate through nodes in depth first search (DFS) post-order.

            :param root: a node of OMDD.
            :return: a set of nodes in DFS-post-order.
        """

        #####################################################
        def _dfs_postorder(dd, nd, visited):
            if dd.out_degree(nd):
                for chd in dd.successors(nd):
                    yield from _dfs_postorder(dd, chd, visited)
            if nd not in visited:
                visited.add(nd)
                yield nd

        #####################################################
        yield from _dfs_postorder(self.graph, root, set())

    def model_counting(self, data_pt, tar, univ):
        """
            BE CAREFUL about the MULTI-EDGES!
            Given a list of universal features,
            count the number of models and universal features.

            :paran data_pt: a data point.
            :param tar: target value of the given data point.
            :param univ: a list of universal features.
            :return: number of models and universal features
        """

        assert len(univ) == self.nf
        assign = dict()
        G = self.graph
        for nd in self.dfs_postorder(self.root):
            if not G.out_degree(nd):
                if G.nodes[nd]['target'] == tar:
                    assign.update({nd: 1})
                else:
                    assign.update({nd: 0})
        for nd in self.dfs_postorder(self.root):
            if not G.out_degree(nd):
                continue
            feat_nd = G.nodes[nd]['var']
            f_id_nd = self.features.index(feat_nd)
            feat_lvl_nd = self.feat2lvl[feat_nd]
            total = 0
            if univ[f_id_nd]:
                for s in G.successors(nd):
                    assert s in assign
                    if G.out_degree(s):
                        feat_s = G.nodes[s]['var']
                        feat_lvl_s = self.feat2lvl[feat_s]
                    else:
                        feat_lvl_s = self.nf
                    assert feat_lvl_nd < feat_lvl_s
                    prod = 1
                    for lvl_i in range(feat_lvl_nd+1, feat_lvl_s):
                        f_i = self.features.index(self.lvl2feat[lvl_i])
                        if univ[f_i]:
                            prod *= len(self.feat_domain[self.lvl2feat[lvl_i]])
                    # multi-edges between nd and s
                    total += assign[s] * prod * G.number_of_edges(nd, s)
            else:
                for s in G.successors(nd):
                    assert s in assign
                    # multi-edges case is not appliable
                    if tuple((nd, s, data_pt[f_id_nd])) in G.edges:
                        if G.out_degree(s):
                            feat_s = G.nodes[s]['var']
                            feat_lvl_s = self.feat2lvl[feat_s]
                        else:
                            feat_lvl_s = self.nf
                        assert feat_lvl_nd < feat_lvl_s
                        prod = 1
                        for lvl_i in range(feat_lvl_nd+1, feat_lvl_s):
                            f_i = self.features.index(self.lvl2feat[lvl_i])
                            if univ[f_i]:
                                prod *= len(self.feat_domain[self.lvl2feat[lvl_i]])
                        total += assign[s] * prod
            assert nd not in assign
            assign.update({nd: total})
        assert self.root in assign
        n_model = assign[self.root]
        return n_model
