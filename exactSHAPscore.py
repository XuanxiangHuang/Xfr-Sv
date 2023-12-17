#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Compute the exact SHAP scores using Barcelo's algorithm (under uniform distribution)
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
from classifiers.ddnnf import dDNNF
from classifiers.omdd import OMDD
from SHAPddnnf import SHAPdDNNF
from SHAPmdd import SHAPoMDD
################################################################################


if __name__ == '__main__':
    
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == '-bench':
        bench_name = args[1]
        fld = args[2]

        with open(bench_name, 'r') as fp:
            datasets = fp.readlines()

        if fld in ["mdds", "dts"]:
            for ds in datasets:
                name = ds.strip()
                data = f"samples/{fld}/{name}.csv"
                _name = name.replace("-modified", "")
                mdd_file = f"models/{fld}/{_name}.mdd"
                print(f"################## {name} ##################")
                df = pd.read_csv(data)
                features = list(df.columns)
                target = features.pop()
                Xs = df[features].values.astype(int)
                mdd_model = OMDD.from_file(mdd_file)
                mdd_model.set_fv_probs_uniform()

                scores = []
                for i, x in enumerate(Xs):
                    data_pt = list(x)
                    sv_mdd = SHAPoMDD(mdd=mdd_model, verb=1)
                    feats_score = [None] * mdd_model.nf
                    for feat in range(mdd_model.nf):
                        feats_score[feat] = sv_mdd.algo_by_def(data_pt, feat)
                    score = np.round(feats_score, decimals=4)
                    scores.append(score)
                exact_shap_scores = np.array(scores)
                header_line = ",".join(mdd_model.features)
                header_line = header_line.lstrip("#")
                np.savetxt(f"shap_scores/barcelo/{name}.csv", exact_shap_scores, fmt="%.4f",
                           delimiter=",", header=header_line, comments="")
            
        elif fld == "ddnnfs":
            for ds in datasets:
                name = ds.strip()
                print(f"################## {name} ##################")
                ddnnf = dDNNF.from_file(f'models/{fld}/{name}/ddnnf_2_fanin/{name}.dnnf', verb=0)
                ddnnf.parse_feature_map(f'models/{fld}/{name}/{name}.map')
                assert ddnnf.is_smooth()
                assert ddnnf.is_decomposable()
                df_X = pd.read_csv(f"samples/{fld}/{name}.csv")
                feature_names = list(df_X.columns)

                prior_distrubution = [0.5] * ddnnf.nf
                shapddnnf = SHAPdDNNF(prior_distrubution)
                scores = []
                for idx, line in enumerate(df_X.to_numpy()):
                    ddnnf.parse_data_point(list(line))
                    feats_score = [None] * ddnnf.nf
                    for feat in range(ddnnf.nf):
                        feats_score[feat] = shapddnnf.algo1(ddnnf, feat)
                    scores.append(feats_score)
                exact_shap_scores = np.array(scores)
                header_line = ",".join(feature_names)
                header_line = header_line.lstrip("#")
                np.savetxt(f"shap_scores/barcelo/{name}.csv", exact_shap_scores, delimiter=",", header=header_line, comments="")
        