#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   logic explainer
#   Author: Xuanxiang Huang
#
################################################################################
from abc import ABC, abstractmethod
import time
from pysat.formula import IDPool
from pysat.solvers import Solver as SAT_Solver
################################################################################


class LogicXplainer(ABC):
    def __init__(self, custom_object, verbose=1):
        self.custom_object = custom_object
        self.verbose = verbose

    @abstractmethod
    def waxp(self, custom_object, fixed):
        """
            User-defined procedure waxp.
            Should test the custom_object and return the result.
        """
        pass

    @abstractmethod
    def wcxp(self, custom_object, universal):
        """
            User-defined procedure waxp.
            Should test the custom_object and return the result.
        """
        pass

    def axp_del(self, fixed):
        """
            Compute one abductive explanation (Axp) using deletion based algorithm.

            :param fixed: a list of features declared as fixed.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """

        time_start = time.perf_counter()

        fix = fixed[:]
        for i in fix:
            tmp_fix = fix[:]
            tmp_fix.remove(i)
            if self.waxp(self.custom_object, tmp_fix):
                fix = tmp_fix
        axp = fix

        time_end = time.perf_counter()

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp (Del): {axp}")
            print("Runtime: {0:.3f}".format(time_end - time_start))

        return axp

    def axp_qxp(self, fixed):
        """
            Compute one abductive explanation (Axp) using divide-and-conquer.
            (QuickExplain algorithm)
            :param fixed: a list of features declared as fixed.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """

        def qxp_recur(B, Z, newB=False):
            if newB and self.waxp(self.custom_object, B):
                return []
            if len(Z) == 1:
                return Z
            u = int(len(Z) / 2)
            Z1 = Z[:u]
            Z2 = Z[u:]
            Q2 = qxp_recur(B + Z1, Z2, len(Z1) > 0)
            Q1 = qxp_recur(B + Q2, Z1, len(Q2) > 0)
            return Q1 + Q2

        time_start = time.perf_counter()

        axp = qxp_recur([], fixed, False)

        time_end = time.perf_counter()

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp (Qxp): {axp}")
            print("Runtime: {0:.3f}".format(time_end - time_start))

        return axp

    def cxp_del(self, universal):
        """
            Compute one contrastive explanation (Cxp) using deletion based algorithm.

            :param universal: a list of features declared as universal.
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """

        time_start = time.perf_counter()

        univ = universal[:]
        for i in univ:
            tmp_univ = univ[:]
            tmp_univ.remove(i)
            if self.wcxp(self.custom_object, tmp_univ):
                univ = tmp_univ
        cxp = univ

        time_end = time.perf_counter()

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp (Del): {cxp}")
            print("Runtime: {0:.3f}".format(time_end - time_start))

        return cxp

    def enum(self, feats_idx, alg='del'):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.
            :param feats_idx: set of feature indices
            :param alg: algorithm used to compute one explanation, 'del' or 'qxp'
            :return: a list of all Axps, a list of all Cxps.
        """

        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        time_start = time.perf_counter()

        axps = []
        cxps = []

        for i in feats_idx:
            new_var(f'u_{i}')

        with SAT_Solver(name="glucose4") as slv:
            while slv.solve():
                # first model is empty
                model = slv.get_model()
                univ = []
                for lit in model:
                    name = vpool.obj(abs(lit)).split(sep='_')
                    univ.extend([int(name[1])] if lit > 0 else [])  # lit > 0 means universal
                fix = [i for i in feats_idx if i not in univ]
                if self.wcxp(self.custom_object, univ):
                    cxp = self.cxp_del(univ)
                    # fix one feature next time
                    slv.add_clause([-new_var(f'u_{i}') for i in cxp])
                    cxps.append(cxp)
                else:
                    if alg == 'del':
                        axp = self.axp_del(fix)
                    elif alg == 'qxp':
                        axp = self.axp_qxp(fix)
                    # free one feature next time
                    slv.add_clause([new_var(f'u_{i}') for i in axp])
                    axps.append(axp)

        time_end = time.perf_counter()
        if self.verbose:
            print('#AXp:', len(axps))
            print('#CXp:', len(cxps))
            print("Runtime: {0:.3f}".format(time_end - time_start))

        return axps, cxps

    def check_axp(self, axp):
        """
            Check if given axp is 1) a weak AXp and 2) subset-minimal.

            :param axp: given axp.
            :return: true if given axp is an AXp
                        else false.
        """

        fix = axp[:]
        # 1) a weak AXp ?
        if not self.waxp(self.custom_object, fix):
            print(f'{axp} is not a weak AXp')
            return False
        # 2) subset-minimal ?
        for i in fix:
            tmp_fix = fix[:]
            tmp_fix.remove(i)
            if self.waxp(self.custom_object, tmp_fix):
                print(f'{axp} is not subset-minimal')
                return False
        return True

    def check_cxp(self, cxp):
        """
            Check if given cxp is 1) a weak CXp and 2) subset-minimal.

            :param cxp: given cxp.
            :return: true if given cxp is an CXp
                        else false.
        """

        univ = cxp[:]
        # 1) a weak CXp ?
        if not self.wcxp(self.custom_object, univ):
            print(f'{cxp} is not a weak CXp')
            return False
        # 2) subset-minimal ?
        for i in univ:
            tmp_univ = univ[:]
            tmp_univ.remove(i)
            if self.wcxp(self.custom_object, tmp_univ):
                print(f'{cxp} is not subset-minimal')
                return False
        return True
