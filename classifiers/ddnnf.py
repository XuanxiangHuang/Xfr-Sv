#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   deterministic Decomposable Negation Normal Form (d-DNNF) circuit
#   Author: Xuanxiang Huang
#
################################################################################
import numpy as np
import pandas as pd
import networkx as nx
################################################################################


class dDNNF(object):
    """
        d-DNNF classifier.
    """
    def __init__(self, nn, nnf, root, leafs, l2l, verb=0):
        self.nn = nn            # num of nodes
        self.nnf = nnf          # nnf graph
        self.root = root        # root node
        self.leafs = leafs      # leaf nodes
        self.lit2leaf = l2l     # map a literal to its corresponding leaf node
        self.nf = None          # num of features
        self.feats = None       # features
        self.domtype = None     # type of domain ('discrete', 'continuous')
        self.bfeats = None      # binarized features (grouped together)
        self.bflits = None      # literal of binarized feature (grouped together)
        self.lits = None        # literal converted from input data point
        self.verbose = verb     # verbose level

    @classmethod
    def from_file(cls, filename, verb=0):
        """
            Load (smooth) d-DNNF model from .ddnnf format file.
            Given .ddnnf file MUST CONTAIN a smooth d-DNNF model.

            :param filename: file in .ddnnf format.
            :param verb: verbose level
            :return: (smooth) d-DNNF model.
        """

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        # filtering out comment lines (those that start with '#')
        lines = list(filter(lambda l: (not (l.startswith('#') or l.strip() == '')), lines))

        lit2leaf = dict()
        leaf = []
        lits = []
        t_nds = []
        nt_nds = []
        edges = []
        index = 0

        assert (lines[index].strip().startswith('NN:'))
        n_nds = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('NV:'))
        n_vars = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('Root:'))
        root = (lines[index].strip().split())[1]
        index += 1

        assert (lines[index].strip().startswith('TDef:'))
        index += 1

        while not lines[index].strip().startswith('NTDef:'):
            nd, t = lines[index].strip().split()
            leaf.append(int(nd))
            if t == 'F' or t == 'T':
                t_nds.append(tuple((int(nd), {'label': t})))
                lit2leaf.update({t: int(nd)})
            else:
                t_nds.append(tuple((int(nd), {'label': int(t)})))
                lit2leaf.update({int(t): int(nd)})
                if abs(int(t)) not in lits:
                    lits.append(abs(int(t)))
            index += 1

        assert (lines[index].strip().startswith('NTDef:'))
        index += 1

        while index < len(lines):
            string = lines[index].strip().split()
            nd = string[0]
            n_type = string[1]
            assert n_type in ('OR', 'AND')
            nt_nds.append(tuple((int(nd), {'label': n_type})))
            chds = string[3:]
            assert len(chds) == int(string[2])
            for chd in chds:
                edges.append(tuple((int(nd), int(chd))))
            index += 1

        assert (len(t_nds) + len(nt_nds)) == int(n_nds)
        assert len(lits) == int(n_vars)

        G = nx.DiGraph()
        G.add_nodes_from(t_nds)
        G.add_nodes_from(nt_nds)
        G.add_edges_from(edges)

        return cls(int(n_nds), G, int(root), leaf, lit2leaf, verb)

    def parse_feature_map(self, map_file):
        """
            Parsing a file mapping a tuple feature,operator,value to a Boolean literal.
            Format is feature:opterator1value(s)(operator2):literal index (>0 or <0).
            operator can be '=', '!=', set '{}', interval ')(]['.

            :param map_file: e.g. age:=12:1 which means literal x_1 denotes age = 12;
                            e.g. age:{10,11,13}:-2 which means literal -x_2 denotes age in {10,11,13};
                            e.g. age:[12,14):3 which means literal x_3 denotes 12<=age<14.
            :return: number of features, features, domain type, binarized features, literals.
        """
        with open(map_file, 'r') as fp:
            lines = fp.readlines()
        # filtering out comment lines (those that start with '#')
        lines = list(filter(lambda l: (not (l.startswith('#') or l.strip() == '')), lines))

        feats = []
        bfeats = []
        bflits = []
        opt_prefix = ('=', '!=', '{', '[', '(')
        index = 0

        assert (lines[index].strip().startswith('NF:'))
        nf = int((lines[index].strip().split())[1])
        index += 1

        assert (lines[index].startswith('Type:'))
        index += 1

        domtype = lines[index].strip().split(',')
        for ele in domtype:
            assert ele in ('discrete', 'continuous')
        index += 1

        assert (lines[index].startswith('Map:'))
        index += 1

        while index < len(lines):
            feat_opt_val_lit = lines[index].strip().split(sep=':')
            dom = feat_opt_val_lit[:-1][-1]
            assert dom.startswith(opt_prefix)
            if dom.startswith('{'):
                assert dom.endswith('}')
            elif dom.startswith(('[', '(')):
                assert dom.endswith((']', ')'))

            if feat_opt_val_lit[0] not in feats:
                feats.append(feat_opt_val_lit[0])
                bfeats.append([tuple(feat_opt_val_lit[:-1])])
                bflits.append([int(feat_opt_val_lit[-1])])

            else:
                idx = feats.index(feat_opt_val_lit[0])
                bfeats[idx].append(tuple(feat_opt_val_lit[:-1]))
                bflits[idx].append(int(feat_opt_val_lit[-1]))

            index += 1

        assert len(feats) == nf
        assert len(bfeats) == nf
        assert len(bflits) == nf
        assert len(domtype) == nf
        self.nf, self.feats, self.domtype, self.bfeats, self.bflits = nf, feats, domtype, bfeats, bflits

        if self.verbose == 2:
            print(f"##### parse feature map #####")
            for f, v, l, dtype in zip(self.feats, self.bfeats, self.bflits, self.domtype):
                print(f"feat: {f}, val: {v}, lit: {l}, type: {dtype}")

    def parse_data_point(self, data_point):
        """
            Parse a data point.
            This is MANDATORY before explaining an instance.
        """
        nf, domtype, bfeats, bflits = self.nf, self.domtype, self.bfeats, self.bflits
        assert (nf == len(data_point))
        lits = []
        for j in range(nf):
            blits = []
            if domtype[j] == 'discrete':
                val_j = str(data_point[j])
                for jj in range(len(bfeats[j])):
                    dom = bfeats[j][jj][1]
                    if dom.startswith('='):
                        if val_j == dom[1:]:
                            blits.append(bflits[j][jj])
                        else:
                            blits.append(-bflits[j][jj])
                    elif dom.startswith('!='):
                        if val_j != dom[2:]:
                            blits.append(bflits[j][jj])
                        else:
                            blits.append(-bflits[j][jj])
                    elif dom.startswith('{'):
                        if val_j in dom[1:-1].split(sep=','):
                            blits.append(bflits[j][jj])
                        else:
                            blits.append(-bflits[j][jj])
            else:
                val_j = float(data_point[j])
                for jj in range(len(bfeats[j])):
                    bound = bfeats[j][jj][1].split(',')
                    lbound = float(bound[0][1:])
                    ubound = float(bound[1][:-1])
                    if bound[0].startswith('(') and bound[1].endswith(')') and lbound < val_j < ubound:
                        blits.append(bflits[j][jj])
                    elif bound[0].startswith('[') and bound[1].endswith(']') and lbound <= val_j <= ubound:
                        blits.append(bflits[j][jj])
                    elif bound[0].startswith('(') and bound[1].endswith(']') and lbound < val_j <= ubound:
                        blits.append(bflits[j][jj])
                    elif bound[0].startswith('[') and bound[1].endswith(')') and lbound <= val_j < ubound:
                        blits.append(bflits[j][jj])
                    else:
                        blits.append(-bflits[j][jj])

            # all literals are consistent.
            for ele in blits:
                assert -ele not in blits
            # no literal occur more than once
            tmp = list(set(blits))
            tmp.sort(key=abs)
            lits.append(tmp)

        assert len(lits) == nf
        self.lits = lits

    def dfs_postorder(self, root):
        """
            Iterate through nodes in depth first search (DFS) post-order.

            :param root: a node of d-DNNF.
            :return: a set of nodes in DFS-post-order.
        """

        #####################################################
        def _dfs_postorder(circuit, nd, visited):
            if circuit.out_degree(nd):
                for chd in circuit.successors(nd):
                    yield from _dfs_postorder(circuit, chd, visited)
            if nd not in visited:
                visited.add(nd)
                yield nd
        #####################################################
        yield from _dfs_postorder(self.nnf, root, set())

    def get_prediction(self):
        """
            Return prediction of lits (which corresponds to the given data point).

            :return:
        """
        circuit = self.nnf
        assign = dict()
        for leaf in self.leafs:
            if circuit.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif circuit.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})
        for lit in self.lits:
            if lit:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if circuit.nodes[nd]['label'] == 'AND'\
                    or circuit.nodes[nd]['label'] == 'OR':
                tmp = [assign[chd] for chd in circuit.successors(nd)]
                if circuit.nodes[nd]['label'] == 'AND':
                    if 0 in tmp:
                        assign.update({nd: 0})
                    else:
                        assign.update({nd: 1})
                else:
                    if 1 in tmp:
                        assign.update({nd: 1})
                    else:
                        assign.update({nd: 0})

        assert assign[self.root] == 1 or assign[self.root] == 0
        return assign[self.root]

    def is_decomposable(self):
        """
            Check if d-DNNF is decomposable

            :return: True if decomposable
        """
        circuit = self.nnf
        scope = dict()
        for leaf in self.leafs:
            if circuit.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif circuit.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = circuit.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in circuit.successors(nd)]
                if circuit.nodes[nd]['label'] == 'AND':
                    for i in range(len(chd_var)):
                        for j in range(i + 1, len(chd_var)):
                            if not chd_var[i].isdisjoint(chd_var[j]):
                                return False
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return True

    def is_smooth(self):
        """
            Check if d-DNNF is smooth

            :return: True if smooth
        """
        circuit = self.nnf
        scope = dict()
        for leaf in self.leafs:
            if circuit.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif circuit.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = circuit.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in circuit.successors(nd)]
                if circuit.nodes[nd]['label'] == 'OR':
                    for i in range(len(chd_var)):
                        for j in range(i + 1, len(chd_var)):
                            if chd_var[i] != chd_var[j]:
                                return False
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return True

    def vars_of_gates(self):
        """
            Collect vars for each gate.

            :return:
        """
        circuit = self.nnf
        scope = dict()
        for leaf in self.leafs:
            if circuit.nodes[leaf]['label'] == 'F':
                scope.update({leaf: frozenset()})
            elif circuit.nodes[leaf]['label'] == 'T':
                scope.update({leaf: frozenset()})
            else:
                lit = circuit.nodes[leaf]['label']
                scope.update({leaf: frozenset({abs(lit)})})

        for nd in self.dfs_postorder(self.root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                chd_var = [scope.get(chd) for chd in circuit.successors(nd)]
                tmp = frozenset()
                for ele in chd_var:
                    tmp = tmp.union(ele)
                scope.update({nd: tmp})
        return scope

    def predict(self, data_points):
        """
            Return a list of prediction given a list of data points.
            :param data_points: a list of (total) data point.
            :return: predictions of these data points
        """
        dpts = data_points
        if type(data_points) == pd.DataFrame:
            dpts = data_points.to_numpy()
        predictions = []
        for dpt in dpts:
            self.parse_data_point([int(e) for e in list(dpt)])
            predictions.append(self.get_prediction())
        return np.array(predictions)

    def model_counting(self, univ):
        """
            Given a list of universal features, return the number of models.

            :param univ: a list of universal features.
            :return: number of models
        """
        nnf = self.nnf
        assign = dict()
        n_univ_var = 0

        for leaf in self.leafs:
            if nnf.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif nnf.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})

        for i in range(self.nf):
            lit = self.lits[i]
            if i in univ:
                for ele in lit:
                    if ele in self.lit2leaf or -ele in self.lit2leaf:
                        n_univ_var += 1
            else:
                for ele in lit:
                    if ele in self.lit2leaf:
                        assign.update({self.lit2leaf[ele]: 1})
                    if -ele in self.lit2leaf:
                        assign.update({self.lit2leaf[-ele]: 0})

        for leaf in self.leafs:
            if leaf not in assign:
                assign.update({leaf: 1})

        assert len(assign) == len(self.leafs)

        for nd in self.dfs_postorder(self.root):
            if nnf.nodes[nd]['label'] == 'AND' \
                    or nnf.nodes[nd]['label'] == 'OR':
                if nnf.nodes[nd]['label'] == 'AND':
                    num = 1
                    for chd in nnf.successors(nd):
                        num *= assign[chd]
                    assign.update({nd: num})
                else:
                    num = 0
                    for chd in nnf.successors(nd):
                        num += assign[chd]
                    assign.update({nd: num})

        n_model = assign[self.root]
        assert n_univ_var >= 0
        return n_model
