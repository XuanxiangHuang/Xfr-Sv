#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   OMDD classifiers explainer
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
from typing import Tuple, List, Any
from Xplainer.LXp import LogicXplainer
from classifiers.omdd import OMDD
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


class XpOMDD(LogicXplainer):
    def __init__(self, custom_object, verb):
        super().__init__(custom_object, verb)

    def waxp(self, custom_object: Tuple[OMDD, List, Any, Any], fixed) -> bool:
        dd, feats_idx, data_point, prediction = custom_object
        univ = [i for i in feats_idx if i not in fixed]
        if not dd.path_to_other_class(data_point, prediction, univ):
            return True
        return False

    def wcxp(self, custom_object: Tuple[OMDD, List, Any, Any], universal) -> bool:
        dd, feats_idx, data_point, prediction = custom_object
        if dd.path_to_other_class(data_point, prediction, universal):
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
            print(f"############ {name} ############")
            df = pd.read_csv(data)
            features = list(df.columns)
            target = features.pop()
            Xs = df[features].values.astype(int)
            mdd_model = OMDD.from_file(mdd_file)
            nn = len(mdd_model.graph.nodes)
            nf = mdd_model.nf
            feats_id = [i for i in range(nf)]
            assert mdd_model.features == features
            assert mdd_model.target == target

            d_len = len(Xs)
            for i, x in enumerate(Xs):
                print(f"{_name}, {i}-th data point")
                pred = mdd_model.predict([x])
                xpmdd = XpOMDD((mdd_model, feats_id, list(x), pred), 1)
                axps, cxps = xpmdd.enum(feats_id, 'qxp')
                for item in axps:
                    assert xpmdd.check_axp(item)
                for item in cxps:
                    assert xpmdd.check_cxp(item)
                assert checkMHS(axps, cxps)