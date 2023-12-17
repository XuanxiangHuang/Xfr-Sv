#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using the SHAP tool (under uniform distribution)
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
import shap
from classifiers.ddnnf import dDNNF

np.random.seed(73)
################################################################################

if __name__ == '__main__':
    # string to bytes
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for item in name_list:
            name = item.strip()
            print(f"################## {name} ##################")
            ################## read d-DNNF ##################
            ddnnf = dDNNF.from_file(f'models/ddnnfs/{name}/ddnnf_2_fanin/{name}.dnnf', verb=0)
            ddnnf.parse_feature_map(f'models/ddnnfs/{name}/{name}.map')
            ################## read d-DNNF ##################
            ################## read data ##################
            df_X = pd.read_csv(f"samples/ddnnfs/{name}.csv")
            feature_names = list(df_X.columns)
            ################## read data ##################

            ################## invoke SHAP explainer ##################
            # The d-DNNF models we evaluate have ≤ 10 features, so the explainer will
            # use the 'exact' algorithm (which is model-agnostic) to compute the Shapley values.
            explainer = shap.Explainer(model=ddnnf.predict, masker=df_X, feature_names=feature_names)
            # The values in the i-th column represent the Shapley values of the corresponding i-th feature.
            approx_shap_values = explainer(df_X)
            header_line = ",".join(feature_names)
            header_line = header_line.lstrip("#")
            np.savetxt(f"shap_scores/lundberg/{name}.csv", approx_shap_values.values, delimiter=",", header=header_line, comments="")
