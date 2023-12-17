#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Computing SHAP scores for OMDD classifiers
#   Author: Xuanxiang Huang
#
################################################################################
import math
from itertools import chain, combinations
from classifiers.omdd import OMDD
################################################################################


def powerset_generator(input):
    # Generate all subsets of the input set
    for subset in chain.from_iterable(combinations(input, r) for r in range(len(input) + 1)):
        yield set(subset)


class SHAPoMDD(object):
    """
        Compute SHAP-score of OMDD, note that OMDD support polytime model counting.
    """

    def __init__(self, mdd: OMDD, verb=0):
        self.mdd = mdd
        self.verbose = verb

    def expect_value(self, data_pt, univ):
        """
            Compute the expectation value of the given data point.
        """
        label_cnt = dict()
        for i in self.mdd.tar_range:
            cnt = self.mdd.model_counting(data_pt, i, univ)
            label_cnt.update({i: cnt})
        expect_val = sum(i * label_cnt[i] for i in label_cnt)
        for i in range(self.mdd.nf):
            feat = self.mdd.features[i]
            if univ[i]:
                expect_val *= self.mdd.fv_probs[feat][self.mdd.feat_domain[feat].index(data_pt[i])]
        return expect_val

    def algo_by_def(self, data_pt, target_feat):
        """
            Computing SHAP-score by definition (using model counting).
        :param data_pt: given data point
        :param target_feat: given feature
        :return: the SHAP-score of the target feature on given data point
        with respect to given OMDD under uniform distribution
        """
        nf = self.mdd.nf
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
            mds_with_t = self.expect_value(data_pt, univ)
            univ[target_feat] = True
            mds_without_t = self.expect_value(data_pt, univ)
            if mds_with_t - mds_without_t == 0:
                continue
            shap_score += math.factorial(len_S) * math.factorial(nf-len_S-1) * (mds_with_t - mds_without_t) / math.factorial(nf)
        return shap_score
