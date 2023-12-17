#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   d-DNNF Classifiers explainer
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
from typing import Tuple, List, Any
from classifiers.ddnnf import dDNNF
from Xplainer.LXp import LogicXplainer
################################################################################


def checkMHS(in_axps: list, in_cxps: list):
    # given a list of axp and a list of cxp,
    # check if they are minimal-hitting-set (MHS) of each other
    # 1. uniqueness, and no subset(superset) exists;
    if not in_axps or not in_cxps:
        print(f"input empty: {in_axps}, {in_cxps}")
        return False
    axps = sorted(in_axps, key=lambda x: len(x))
    axps_ = axps[:]
    while axps:
        axp = axps.pop()
        set_axp = set(axp)
        for ele in axps:
            set_ele = set(ele)
            if set_axp.issuperset(set_ele) or set_axp.issubset(set_ele):
                print(f"axp is not unique: {set_axp}, {set_ele}")
                return False
    cxps = sorted(in_cxps, key=lambda x: len(x))
    cxps_ = cxps[:]
    while cxps:
        cxp = cxps.pop()
        set_cxp = set(cxp)
        for ele in cxps:
            set_ele = set(ele)
            if set_cxp.issuperset(set_ele) or set_cxp.issubset(set_ele):
                print(f"cxp is not unique: {set_cxp}, {set_ele}")
                return False
    # 2. minimal hitting set;
    for axp in axps_:
        set_axp = set(axp)
        for cxp in cxps_:
            set_cxp = set(cxp)
            if not (set_axp & set_cxp):  # not a hitting set
                print(f"not a hitting set: axp:{set_axp}, cxp:{set_cxp}")
                return False
    # axp is a MHS of cxps
    for axp in axps_:
        set_axp = set(axp)
        for ele in set_axp:
            tmp = set_axp - {ele}
            size = len(cxps_)
            for cxp in cxps_:
                set_cxp = set(cxp)
                if tmp & set_cxp:
                    size -= 1
            if size == 0:  # not minimal
                print(f"axp is not minimal hitting set: "
                      f"axp {set_axp} covers #{len(cxps_)}, "
                      f"its subset {tmp} covers #{len(cxps_) - size}, "
                      f"so {ele} is redundant")
                return False
    # cxp is a MHS of axps
    for cxp in cxps_:
        set_cxp = set(cxp)
        for ele in set_cxp:
            tmp = set_cxp - {ele}
            size = len(axps_)
            for axp in axps_:
                set_axp = set(axp)
                if tmp & set_axp:
                    size -= 1
            if size == 0:
                print(f"cxp is not minimal hitting set: "
                      f"cxp {set_cxp} covers #{len(axps_)}, "
                      f"its subset {tmp} covers #{len(axps_) - size}, "
                      f"so {ele} is redundant")
                return False
    return True


class XpdDnnf(LogicXplainer):
    """
        Explain d-DNNF classifier.
    """
    def __init__(self, custom_object, verb):
        super().__init__(custom_object, verb)

    def check_ICoVa(self, ddnnf: dDNNF, univ, va=True) -> bool:
        """
            Given a list of universal features, check inconsistency or validity.

            :param univ: a list of universal features.
            :param va: True if check validity else check inconsistency.
            :return: True if pass the check
        """
        circuit = ddnnf.nnf
        assign = dict()
        n_univ_var = 0

        for leaf in ddnnf.leafs:
            if circuit.nodes[leaf]['label'] == 'F':
                assign.update({leaf: 0})
            elif circuit.nodes[leaf]['label'] == 'T':
                assign.update({leaf: 1})

        for i in range(ddnnf.nf):
            lit = ddnnf.lits[i]
            if i in univ:
                for ele in lit:
                    if ele in ddnnf.lit2leaf or -ele in ddnnf.lit2leaf:
                        n_univ_var += 1
            else:
                for ele in lit:
                    if ele in ddnnf.lit2leaf:
                        assign.update({ddnnf.lit2leaf[ele]: 1})
                    if -ele in ddnnf.lit2leaf:
                        assign.update({ddnnf.lit2leaf[-ele]: 0})

        for leaf in ddnnf.leafs:
            if leaf not in assign:
                assign.update({leaf: 1})

        assert len(assign) == len(ddnnf.leafs)

        for nd in ddnnf.dfs_postorder(ddnnf.root):
            if circuit.nodes[nd]['label'] == 'AND' \
                    or circuit.nodes[nd]['label'] == 'OR':
                if circuit.nodes[nd]['label'] == 'AND':
                    num = 1
                    for chd in circuit.successors(nd):
                        num *= assign[chd]
                    assign.update({nd: num})
                else:
                    num = 0
                    for chd in circuit.successors(nd):
                        num += assign[chd]
                    assign.update({nd: num})

        n_model = assign[ddnnf.root]
        assert n_univ_var >= 0

        if va:
            return n_model == 2 ** n_univ_var
        else:
            return n_model == 0

    def waxp(self, custom_object: Tuple[dDNNF, List, Any], fixed) -> bool:
        circuit, feats_idx, prediction = custom_object
        univ = [i for i in feats_idx if i not in fixed]
        if (prediction and self.check_ICoVa(circuit, univ, va=True)) or \
                (not prediction and self.check_ICoVa(circuit, univ, va=False)):
            return True
        return False

    def wcxp(self, custom_object: Tuple[dDNNF, List, Any], universal) -> bool:
        circuit, feats_idx, prediction = custom_object
        if (prediction and not self.check_ICoVa(circuit, universal, va=True)) or \
                (not prediction and not self.check_ICoVa(circuit, universal, va=False)):
            return True
        return False


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == '-bench':
        bench_name = args[1]
        fld = args[2]

        with open(bench_name, 'r') as fp:
            datasets = fp.readlines()

        for ds in datasets:
            name = ds.strip()
            data = f"samples/{fld}/{name}.csv"
            _name = name.replace("-modified", "")
            mdd_file = f"models/{fld}/{_name}.mdd"
            print(f"################## {name} ##################")
            ddnnf = dDNNF.from_file(f'models/{fld}/{name}/ddnnf_2_fanin/{name}.dnnf', verb=1)
            ddnnf.parse_feature_map(f'models/{fld}/{name}/{name}.map')
            df = pd.read_csv(data)
            features = list(df.columns)
            Xs = df[features].values.astype(int)
            nf = ddnnf.nf
            feats_id = [i for i in range(nf)]

            d_len = len(Xs)
            for i, x in enumerate(Xs):
                print(f"{_name}, {i}-th instance")
                ddnnf.parse_data_point(list(x))
                pred = ddnnf.get_prediction()
                xpddnnf = XpdDnnf((ddnnf, feats_id, pred), 1)
                axps, cxps = xpddnnf.enum(feats_id, 'del')
                for item in axps:
                    assert xpddnnf.check_axp(item)
                for item in cxps:
                    assert xpddnnf.check_cxp(item)
                assert checkMHS(axps, cxps)
