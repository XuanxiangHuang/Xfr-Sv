#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   SHAP score of (explanation) irrelevant feature > SHAP score of (explanation) relevant feature
#   Author: Xuanxiang Huang
#
################################################################################
import sys
import pandas as pd
import numpy as np
from classifiers.omdd import OMDD
from classifiers.ddnnf import dDNNF
from LXp_ddnnf import XpdDnnf
from LXp_mdd import XpOMDD
import matplotlib.pyplot as plt
################################################################################

# The palette with grey:
# "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"


def plot_max_irr_min_rel(data, col_names, fig_title, filename, score_type="lundberg"):
    df = pd.DataFrame(data, columns=col_names)
    ax1 = df.plot.scatter(x=col_names[0], y=col_names[1], color="#E69F00")
    ax2 = df.plot.scatter(x=col_names[0], y=col_names[2], color="#56B4E9", ax=ax1)
    ax2.set_xlabel("Instance #")
    ax2.set_ylabel("SHAP scores" if score_type == "lundberg" else "exact SHAP scores")  # SHAP or exact SHAP
    # plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


def plot_irr_rel_summary(data, len_X, fig_title, filename, score_type="lundberg"):
    plt.rcdefaults()
    plt.bar(len_X, data, align='center', color=["#CC79A7", "#009E73"])
    plt.annotate(f'{data[0]}', xy=(0, data[0]), ha='center', va='bottom')
    plt.annotate(f'{data[1]}', xy=(1, data[1]), ha='center', va='bottom')
    plt.ylabel('# Instances')
    plt.xlabel('SHAP scores' if score_type == "lundberg" else 'exact SHAP scores')  # SHAP or exact SHAP
    # plt.title(fig_title)
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


# python3 XXX.py -bench pmlb_bool.txt lundberg
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == '-bench':
        bench_name = args[1]
        fld = args[2]
        which_score = args[3]

        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        if fld in ["mdds", "dts"]:
            for item in name_list:
                name = item.strip()
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
                nf = mdd_model.nf
                feats_id = [i for i in range(nf)]

                b_data = pd.read_csv(f"shap_scores/{which_score}/{name}.csv")
                b_score = np.abs(b_data.to_numpy())
                n, m = df.shape
                max_irr_and_min_rel = []
                max_ir_lt_min_r = 0
                max_ir_ge_min_r = 0
                rel_zero = 0
                diff_max_ir_min_r = []
                column_names = ["#Instance", "Max Scores of IR-Feat", "Min Scores of R-Feat"]
                count = 0

                for i, x in enumerate(Xs):
                    pred = mdd_model.predict([x])
                    data_pt = list(x)
                    xpmdd = XpOMDD((mdd_model, feats_id, data_pt, pred), 0)

                    feat_cnts = nf * [0]
                    axps, cxps = xpmdd.enum(feats_id, 'del')
                    for axp in axps:
                        for feat in axp:
                            feat_cnts[feat] += 1

                    scores_irr = [b_score[i, j] for j in range(nf) if feat_cnts[j] == 0]
                    scores_rel = [b_score[i, j] for j in range(nf) if feat_cnts[j] != 0]
                    if len(scores_irr) and len(scores_rel):
                        if max(scores_irr) >= min(scores_rel):
                            max_irr_and_min_rel.append([count, max(scores_irr), min(scores_rel)])
                            diff_max_ir_min_r.append([count, max(scores_irr) - min(scores_rel)])
                            max_ir_ge_min_r += 1
                            count += 1
                        else:
                            max_ir_lt_min_r += 1
                    else:
                        max_ir_lt_min_r += 1

                    if 0 in scores_rel:
                        print(data_pt, pred, b_data.to_numpy()[i], sum(b_data.to_numpy()[i]),
                              feat_cnts, axps)
                        rel_zero += 1

                if max_ir_ge_min_r:
                    plot_max_irr_min_rel(np.array(max_irr_and_min_rel), column_names, name,
                                         f"shap_scores/IrrRel/{which_score}_scores/{name}.png", which_score)
                    plot_irr_rel_summary(np.asarray([max_ir_ge_min_r, max_ir_lt_min_r]), ['Abnormal', 'Normal'], name,
                                         f"shap_scores/IrrRel/{which_score}_summary/{name}.png", which_score)
                else:
                    print("No Sv(Irr) >= Sv(Rel) issues found!")

                if rel_zero:
                    print(f"Rel has zero score in {rel_zero} instances!")

        elif fld == "ddnnfs":
            for item in name_list:
                name = item.strip()
                print(f"################## {name} ##################")
                ################## read d-DNNF ##################
                ddnnf = dDNNF.from_file(f'models/{fld}/{name}/ddnnf_2_fanin/{name}.dnnf', verb=0)
                ddnnf.parse_feature_map(f'models/{fld}/{name}/{name}.map')
                ################## read d-DNNF ##################
                df_X = pd.read_csv(f"samples/{fld}/{name}.csv")
                feature_names = list(df_X.columns)
                feats_id = [i for i in range(ddnnf.nf)]

                b_data = pd.read_csv(f"shap_scores/{which_score}/{name}.csv")
                b_score = np.round(np.abs(b_data.to_numpy()), decimals=4)
                n, m = df_X.shape
                max_irr_and_min_rel = []
                max_ir_lt_min_r = 0
                max_ir_ge_min_r = 0
                diff_max_ir_min_r = []
                column_names = ["#Instance", "Max Scores of IR-Feat", "Min Scores of R-Feat"]
                count = 0
                for idx, line in enumerate(df_X.to_numpy()):
                    ddnnf.parse_data_point(list(line))
                    feat_cnts = [0] * ddnnf.nf
                    pred = ddnnf.get_prediction()
                    xpddnnf = XpdDnnf((ddnnf, feats_id, pred), 0)
                    axps, cxps = xpddnnf.enum(feats_id, 'del')
                    for axp in axps:
                        for feat in axp:
                            assert feat < ddnnf.nf
                            feat_cnts[feat] += 1
                    scores_irr = [b_score[idx, j] for j in range(ddnnf.nf) if feat_cnts[j] == 0]
                    scores_rel = [b_score[idx, j] for j in range(ddnnf.nf) if feat_cnts[j] != 0]
                    if len(scores_irr) and len(scores_rel):
                        if max(scores_irr) >= min(scores_rel):
                            max_irr_and_min_rel.append([count, max(scores_irr), min(scores_rel)])
                            diff_max_ir_min_r.append([count, max(scores_irr) - min(scores_rel)])
                            max_ir_ge_min_r += 1
                            count += 1
                        else:
                            max_ir_lt_min_r += 1
                    else:
                        max_ir_lt_min_r += 1
                plot_max_irr_min_rel(np.array(max_irr_and_min_rel), column_names, name,
                                     f"shap_scores/IrrRel/{which_score}_scores/{name}.png",
                                     which_score)
                plot_irr_rel_summary(np.asarray([max_ir_ge_min_r, max_ir_lt_min_r]), ['Abnormal', 'Normal'], name,
                                     f"shap_scores/IrrRel/{which_score}_summary/{name}.png",
                                     which_score)