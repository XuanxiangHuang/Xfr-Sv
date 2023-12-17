#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Computing SHAP scores for d-DNNF classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import math
from itertools import chain, combinations
from classifiers.ddnnf import dDNNF
################################################################################


def powerset_generator(input):
    # Generate all subsets of the input set
    for subset in chain.from_iterable(combinations(input, r) for r in range(len(input) + 1)):
        yield set(subset)


class SHAPdDNNF(object):
    """
        Compute SHAP-score of d-DNNF, based on the paper:
        On the Complexity of SHAP-Score-Based Explanations Tractability via Knowledge Compilation and Non-Approximability Results
        (https://arxiv.org/abs/2104.08015)
    """

    def __init__(self, prior_feat_distribution):
        # prior fully-factorized distributions (e.g. Uniform Distribution, Product Distribution)
        self.prior_feat_distribution = prior_feat_distribution

    def algo1(self, ddnnf: dDNNF, target_feat):
        """
            Applicable to smoothed circuit
        :param ddnnf: d-DNNF circuit
        :param target_feat: given feature
        :return: the SHAP-score of the target feature on given data point
        with respect to given d-DNNF circuit under given distribution
        """
        circuit = ddnnf.nnf
        nf = ddnnf.nf
        lits = ddnnf.lits
        root = ddnnf.root
        shap_score = 0
        gamma = dict()
        delta = dict()
        ########## preprocess ##########
        scope = ddnnf.vars_of_gates()
        lit2feat = dict()
        for i in range(nf):
            lit = ddnnf.lits[i][0]
            lit2feat.update({lit: i})
            lit2feat.update({-lit: i})
        ########## preprocess ##########
        # feature range from [0,...,n-1] but var range from [1,...,n]
        target_var = abs(lits[target_feat][0])
        for leaf in ddnnf.leafs:
            # constant gate
            if circuit.nodes[leaf]['label'] == 'F':
                gamma.update({(leaf, 0): 0})
                delta.update({(leaf, 0): 0})
            elif circuit.nodes[leaf]['label'] == 'T':
                gamma.update({(leaf, 0): 1})
                delta.update({(leaf, 0): 1})
            else:
                # variable gate (we only consider boolean domain)
                if target_var in scope.get(leaf):
                    if circuit.nodes[leaf]['label'] > 0:
                        gamma.update({(leaf, 0): 1})
                        delta.update({(leaf, 0): 0})
                    else:
                        gamma.update({(leaf, 0): 0})
                        delta.update({(leaf, 0): 1})
                else:
                    feat = lit2feat.get(circuit.nodes[leaf]['label'])
                    if circuit.nodes[leaf]['label'] > 0:
                        gamma.update({(leaf, 0): self.prior_feat_distribution[feat]})
                        delta.update({(leaf, 0): self.prior_feat_distribution[feat]})
                        gamma.update({(leaf, 1): 1 if lits[feat][0] > 0 else 0})
                        delta.update({(leaf, 1): 1 if lits[feat][0] > 0 else 0})
                    else:
                        gamma.update({(leaf, 0): 1 - self.prior_feat_distribution[feat]})
                        delta.update({(leaf, 0): 1 - self.prior_feat_distribution[feat]})
                        gamma.update({(leaf, 1): 0 if lits[feat][0] > 0 else 1})
                        delta.update({(leaf, 1): 0 if lits[feat][0] > 0 else 1})

        for nd in ddnnf.dfs_postorder(root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                chds = [_ for _ in circuit.successors(nd)]
                assert len(chds) == 2
                var_g = set(scope.get(nd))
                if target_var in var_g:
                    var_g.remove(target_var)
                if circuit.nodes[nd]['label'] == 'OR':
                    for i in range(len(var_g) + 1):
                        tmp_gamma = gamma.get((chds[0], i)) + gamma.get((chds[1], i))
                        gamma.update({(nd, i): tmp_gamma})
                        tmp_delta = delta.get((chds[0], i)) + delta.get((chds[1], i))
                        delta.update({(nd, i): tmp_delta})
                else:
                    var_g0 = set(scope.get(chds[0]))
                    var_g1 = set(scope.get(chds[1]))
                    if target_var in var_g0:
                        var_g0.remove(target_var)
                    if target_var in var_g1:
                        var_g1.remove(target_var)
                    for i in range(len(var_g) + 1):
                        lst0 = list(range(0, min(i, len(var_g0)) + 1))
                        lst1 = list(range(0, min(i, len(var_g1)) + 1))
                        comb = [(l1, l2) for l1 in lst0 for l2 in lst1 if l1 + l2 == i]
                        tmp_gamma = 0
                        tmp_delta = 0
                        for pair in comb:
                            tmp_gamma += gamma.get((chds[0], pair[0])) * gamma.get((chds[1], pair[1]))
                            tmp_delta += delta.get((chds[0], pair[0])) * delta.get((chds[1], pair[1]))
                        gamma.update({(nd, i): tmp_gamma})
                        delta.update({(nd, i): tmp_delta})

        for k in range(nf):
            e = 1 if lits[target_feat][0] > 0 else 0
            e_p = e - self.prior_feat_distribution[target_feat]
            gamma_delta = 0
            if (root, k) in gamma:
                gamma_delta = gamma.get((root, k)) - delta.get((root, k))
            if gamma_delta == 0:
                continue
            shap_score += math.factorial(k) * math.factorial(nf - k - 1) * e_p * gamma_delta
        return shap_score / math.factorial(nf)

    def algo2(self, ddnnf: dDNNF, target_feat):
        """
            Applicable to non-smoothed circuit
        :param ddnnf: d-DNNF explainer (based on formal method)
        :param target_feat: given feature
        :return: the SHAP-score of the target feature on given data point
        with respect to given d-DNNF circuit under given distribution
        """
        circuit = ddnnf.nnf
        nf = ddnnf.nf
        lits = ddnnf.lits
        root = ddnnf.root
        shap_score = 0
        gamma = dict()
        delta = dict()
        ########## preprocess ##########
        scope = ddnnf.vars_of_gates()
        lit2feat = dict()
        for i in range(nf):
            lit = ddnnf.lits[i][0]
            lit2feat.update({lit: i})
            lit2feat.update({-lit: i})
        ########## preprocess ##########
        # feature range from [0,...,n-1] but var range from [1,...,n]
        target_var = abs(lits[target_feat][0])
        for leaf in ddnnf.leafs:
            # constant gate
            if circuit.nodes[leaf]['label'] == 'F':
                gamma.update({(leaf, 0): 0})
                delta.update({(leaf, 0): 0})
            elif circuit.nodes[leaf]['label'] == 'T':
                gamma.update({(leaf, 0): 1})
                delta.update({(leaf, 0): 1})
            else:
                # variable gate (we only consider boolean domain)
                if target_var in scope.get(leaf):
                    if circuit.nodes[leaf]['label'] > 0:
                        gamma.update({(leaf, 0): 1})
                        delta.update({(leaf, 0): 0})
                    else:
                        gamma.update({(leaf, 0): 0})
                        delta.update({(leaf, 0): 1})
                else:
                    feat = lit2feat.get(circuit.nodes[leaf]['label'])
                    if circuit.nodes[leaf]['label'] > 0:
                        gamma.update({(leaf, 0): self.prior_feat_distribution[feat]})
                        delta.update({(leaf, 0): self.prior_feat_distribution[feat]})
                        gamma.update({(leaf, 1): 1 if lits[feat][0] > 0 else 0})
                        delta.update({(leaf, 1): 1 if lits[feat][0] > 0 else 0})
                    else:
                        gamma.update({(leaf, 0): 1 - self.prior_feat_distribution[feat]})
                        delta.update({(leaf, 0): 1 - self.prior_feat_distribution[feat]})
                        gamma.update({(leaf, 1): 0 if lits[feat][0] > 0 else 1})
                        delta.update({(leaf, 1): 0 if lits[feat][0] > 0 else 1})

        for nd in ddnnf.dfs_postorder(root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                chds = [_ for _ in circuit.successors(nd)]
                assert len(chds) == 2
                var_g = set(scope.get(nd))
                if target_var in var_g:
                    var_g.remove(target_var)
                if circuit.nodes[nd]['label'] == 'OR':
                    var_g0 = set(scope.get(chds[0]))
                    var_g1 = set(scope.get(chds[1]))
                    if target_var in var_g0:
                        var_g0.remove(target_var)
                    if target_var in var_g1:
                        var_g1.remove(target_var)
                    var_g0_var_g1 = var_g0 - var_g1
                    var_g1_var_g0 = var_g1 - var_g0
                    assert target_var not in var_g0_var_g1
                    assert target_var not in var_g1_var_g0
                    for i in range(len(var_g) + 1):
                        lst0 = list(range(0, min(i, len(var_g0)) + 1))
                        lst1 = list(range(0, min(i, len(var_g1)) + 1))
                        tmp_gamma = 0
                        tmp_delta = 0
                        for l0 in lst0:
                            coef0 = math.comb(len(var_g1_var_g0), i - l0)
                            tmp_gamma += gamma.get((chds[0], l0)) * coef0
                            tmp_delta += delta.get((chds[0], l0)) * coef0
                        for l1 in lst1:
                            coef1 = math.comb(len(var_g0_var_g1), i - l1)
                            tmp_gamma += gamma.get((chds[1], l1)) * coef1
                            tmp_delta += delta.get((chds[1], l1)) * coef1
                        gamma.update({(nd, i): tmp_gamma})
                        delta.update({(nd, i): tmp_delta})
                else:
                    var_g0 = set(scope.get(chds[0]))
                    var_g1 = set(scope.get(chds[1]))
                    if target_var in var_g0:
                        var_g0.remove(target_var)
                    if target_var in var_g1:
                        var_g1.remove(target_var)
                    for i in range(len(var_g) + 1):
                        lst0 = list(range(0, min(i, len(var_g0)) + 1))
                        lst1 = list(range(0, min(i, len(var_g1)) + 1))
                        comb = [(l1, l2) for l1 in lst0 for l2 in lst1 if l1 + l2 == i]
                        tmp_gamma = 0
                        tmp_delta = 0
                        for pair in comb:
                            tmp_gamma += gamma.get((chds[0], pair[0])) * gamma.get((chds[1], pair[1]))
                            tmp_delta += delta.get((chds[0], pair[0])) * delta.get((chds[1], pair[1]))
                        gamma.update({(nd, i): tmp_gamma})
                        delta.update({(nd, i): tmp_delta})

        for k in range(nf):
            e = 1 if lits[target_feat][0] > 0 else 0
            e_p = e - self.prior_feat_distribution[target_feat]
            gamma_delta = 0
            if (root, k) in gamma:
                gamma_delta = gamma.get((root, k)) - delta.get((root, k))
            if gamma_delta == 0:
                continue
            shap_score += math.factorial(k) * math.factorial(nf - k - 1) * e_p * gamma_delta
        return shap_score / math.factorial(nf)

    def algo_by_def(self, ddnnf: dDNNF, target_feat):
        """
            Computing SHAP-score by definition, applicable to smoothed circuit.
        :param ddnnf: d-DNNF circuit
        :param target_feat: given feature
        :return: the SHAP-score of the target feature on given data point
        with respect to given d-DNNF circuit under uniform distribution ONLY
        """
        nf = ddnnf.nf
        feats = list(range(nf))
        assert target_feat in feats
        feats.remove(target_feat)
        all_S = list(powerset_generator(feats))
        shap_score = 0
        for S in all_S:
            len_S = len(list(S))
            univ = [False] * nf
            for i in range(nf):
                if i not in list(S):
                    univ[i] = True
            univ[target_feat] = False
            mds_with_t = ddnnf.model_counting(univ) * (0.5 ** (nf - len_S - 1))
            univ[target_feat] = True
            mds_without_t = ddnnf.model_counting(univ) * (0.5 ** (nf - len_S))
            if mds_with_t - mds_without_t == 0:
                continue
            shap_score += math.factorial(len_S) * math.factorial(nf - len_S - 1) * \
                          (mds_with_t - mds_without_t) / math.factorial(nf)
        return shap_score
