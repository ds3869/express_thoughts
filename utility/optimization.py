import re
import os
import numpy as np
import pandas as pd
import time
import cvxpy as cp
import winsound
import matplotlib.pyplot as plt
from functools import reduce
from collections import defaultdict
from seaborn import heatmap
from copy import deepcopy
from scipy.optimize import linprog as lp
from cvxopt import solvers, matrix
from utility.single_factor_test import Backtest_stock
from utility.factor_data_preprocess import info_cols
from barra_cne6.barra_template import Data


def select_import_wei(wei_se, n_max, abs_max_ratio=0.9, max_in_indus=2):
    '''
    abs_max = 15        留下绝对数前15的个股权重
    max_in_indus = 2    留下行业内排名前2的个股权重
    同时删除部分权重过小的股票，只留下极少部分股票做进一步的优化，删除第一部优化时权重小于0.001的股票。
    '''
    data = Data()
    basic_inform = data.stock_basic_inform
    indus_map = basic_inform.loc[wei_se.index, '申万一级行业']
    # 强制保留w1中各行业最大权重股以及其他权重前15的股票
    res_wei = pd.Series(index=wei_se.index)

    # 分行业里面权重大的留下
    wei_se = wei_se.sort_index()
    dat_df = pd.DataFrame({'wei': wei_se, 'industry': indus_map})
    grouped = dat_df.groupby('industry')
    for ind, v in grouped:
        tmp = v['wei'].sort_values(ascending=False)
        res_wei[tmp[:max_in_indus].index] = tmp[:max_in_indus]

    num_0 = np.sum(res_wei > 0)
    # 分行业留下权重前2的股票之外，在找到其余的股票中绝对权重排名前 (n_max * abs_max_ratio - 已经留下的股票数量)的股票数量。
    num_1 = int(n_max * abs_max_ratio) - num_0
    tmp = list(set(wei_se.index) - set(res_wei.dropna().index))
    left_se = wei_se[tmp]
    left_se = left_se.sort_values(ascending=False)
    tmp1 = left_se[:num_1]
    res_wei[tmp1.index] = tmp1

    res_wei = res_wei.dropna()
    n_left = np.sum(res_wei > 0)
    # np.sum(res_wei)
    if n_left > n_max:
        print('留下的股票过多，重新选择')
        input("暂时挂起.... ")

    tmp2 = wei_se[wei_se > 0.001]
    tobe_opt = [i for i in tmp2.index if i not in res_wei.index]

    return res_wei, n_left, tobe_opt


# 根据变量和条件，生成约束列
def generates_constraints(para_dict, con_dict):
    '''
        个股权重约束：            1，不为负，nonneg=True
                                 2，最大值, x <= max_wei
        成份股权重和约束（如有）：  is_in_bench.T * x >= in_benchmark_wei
        行业中性约束:             dum @ x == ind_wei,
        行业非中性约束：          dum @ x <= ind_wei + industry_max_expose,
                                 dum @ x >= ind_wei - industry_max_expose,
                                 cp.sum(x) == 1,
        风险因子约束：           limit_df = limit_factor_df[key]
                                # 调整到同一index排序
                                limit_df = limit_df[wei_tmp.index]
                                limit_f = limit_df.T.values
                                bench_limit_expo = np.dot(limit_df.T, wei_tmp)
            1，如果风险因子完全不暴露： limit_f.T * x = bench_limit_expo
            2，风险因子暴露一定数量：   limit_f.T * x >= bench_limit_expo - value
                                       limit_f.T * x <= bench_limit_expo - value
        换手率约束：             cvx.norm(x - pre_x, 1) <= 0.3
        跟踪误差约束：           cp.quad_form(x - wei_tmp, P) <= te
        最大股票数量约束：        y = cp.Variable(len(ret_tmp), boolean=True)
                                 x - y <= 0,
                                 cp.sum(y) <= 120,
        '''

    x = para_dict['x']
    if 'y' in para_dict.keys():
        y = para_dict['y']
        y_sum = para_dict['y_sum']
    else:
        y = None
        y_sum = None

    max_wei = para_dict['max_wei']   # 个股最大权重
    in_benchmark_wei = para_dict['in_benchmark_wei']
    is_in_bench = para_dict['is_in_bench']
    dum = para_dict['dum']
    wei_tmp = para_dict['wei_tmp']
    ind_wei = para_dict['ind_wei']
    ret_e = para_dict['ret_e']
    risk_factor_dict = para_dict['risk_factor_dict']
    limit_factor_df = para_dict['limit_factor_df']
    pre_w = para_dict['pre_w']
    P = para_dict['P']
    total_wei = para_dict['total_wei']

    in_benchmark = con_dict['in_benchmark']
    industry_max_expose = con_dict['industry_max_expose']
    turnover = con_dict['turnover']
    te = con_dict['te']

    constraints = [x <= max_wei]

    if not in_benchmark:
        constraints.append(is_in_bench.values.T * x >= in_benchmark_wei)

    if industry_max_expose == 0:
        constraints.append(dum @ x == ind_wei)
    else:
        constraints.append(dum @ x <= ind_wei + industry_max_expose)
        tmp = ind_wei - industry_max_expose
        tmp[tmp < 0] = 0
        constraints.append(dum @ x >= tmp)
        constraints.append(cp.sum(x) == total_wei)

    if len(risk_factor_dict) != 0:
        for key, value in risk_factor_dict.items():
            print(key)
            print(value)
            limit_df = limit_factor_df[key]
            # 调整到同一index排序
            # 第一次优化时，待优化的x的Index与指数权重wei的index相同，
            # 但第二次优化时，待优化的x的Index与指数权重wei的index不同，算指数的风险因子暴露与算待优化的风险因子暴露
            # 的因子矩阵就不一样了。
            limit_df = limit_df[wei_tmp.index]
            bench_limit_expo = np.dot(limit_df.T, wei_tmp)
            limit_f = limit_df[ret_e.index].T.values

            if y and value == 0:
                value = 2

            if value == 0:
                constraints.append(limit_f.T * x == bench_limit_expo)
            else:
                constraints.append(limit_f.T * x >= bench_limit_expo - value)
                constraints.append(limit_f.T * x <= bench_limit_expo + value)

    if turnover and isinstance(pre_w, pd.Series):
        constraints.append(cp.norm(x - pre_w.values, 1) <= turnover)

    # 当作二次优化的时候，x的数量和指数权重的已经发生很大变化了，而且P也变了。
    # 简化期间，在二次优化时就先忽略这个功能。
    if te and not y:
        constraints.append(cp.quad_form(x - wei_tmp, P) <= te)

    if y:
        constraints.append(x - y <= 0.)
        constraints.append(cp.sum(y) <= y_sum)

    return constraints


def generates_problem(q, x, P, c, pre_w, constraints, te):
    if te:
        if isinstance(pre_w, pd.Series):
            # norm(x, 1) 表示 ∑i|xi|
            prob = cp.Problem(cp.Maximize(q.T * x - c * cp.norm(x-pre_w.values, 1)), constraints)
        else:
            prob = cp.Problem(cp.Maximize(q.T * x), constraints)
    else:
        if isinstance(pre_w, pd.Series):
            prob = cp.Problem(cp.Maximize(q.T * x - cp.quad_form(x, P) - c * cp.norm(x-pre_w.values, 1)), constraints)
        else:
            prob = cp.Problem(cp.Maximize(q.T * x - cp.quad_form(x, P)), constraints)
    return prob


def optimization_fun(ret, e, bench_wei, pre_w=None, is_enhance=True, lamda=10, c=0.015, turnover=None, te=None,
                     industry_max_expose=0, risk_factor_dict={}, limit_factor_df=None, in_benchmark=True,
                     in_benchmark_wei=0.8, max_num=None):
    if in_benchmark:
        # 如果必须在成份股内选择，则需要对风险矩阵进行处理，跳出仅是成份股的子矩阵
        wei_tmp = bench_wei.dropna()
        bug_maybe = [i for i in wei_tmp.index if i not in e.index]
        if len(bug_maybe) > 0:
            print('存在下列股票不在组合里，请检查')
            print(bug_maybe)

        e_tmp = e.loc[wei_tmp.index, wei_tmp.index].fillna(0)
        ret_tmp = ret[wei_tmp.index].fillna(0)
        if pre_w:
            pre_w = pre_w[wei_tmp.index].fillna(0)

    else:
        # 确保几个重要变量有相同的index
        n_index = [i for i in e.index if i in ret.index]
        e_tmp = e.loc[n_index, n_index]
        ret_tmp = ret[n_index]
        wei_tmp = bench_wei[n_index].fillna(0)
        if isinstance(pre_w, pd.Series):
            to_test_list = len([i for i in pre_w.index if i not in n_index])
            if np.any(pre_w[to_test_list] > 0.001):
                input('input:存在部分有权重的股票在上期，而不再当期的数据中，请检查')
            pre_w = pre_w[n_index].fillna(0)
        # 如果可以选非成份股，则可以确定一个成份股权重比例的约束条件。
        is_in_bench = deepcopy(wei_tmp)
        is_in_bench[is_in_bench > 0] = 1       # 代表是否在成份股内的变量

    data = Data()
    basic = data.stock_basic_inform
    industry_sw = basic[['申万一级行业']]
    # 股票组合的行业虚拟变量
    industry_map = industry_sw.loc[ret_tmp.index, :]

    # dummies_bench = pd.get_dummies(industry_map.loc[bench_wei.index, :])
    # dummies_bench.sum()  不同行业的公司数量
    industry_map.fillna('综合', inplace=True)
    dummies = pd.get_dummies(industry_map[industry_map.columns[0]])

    dummies.sum()

    # 个股最大权重为行业权重的 3/4
    ind_wei = np.dot(dummies.T, wei_tmp)
    ind_wei_se = pd.Series(index=dummies.columns, data=ind_wei)
    industry_map['max_wei'] = None
    for i in industry_map.index:
        try:
            industry_map.loc[i, 'max_wei'] = 0.75*ind_wei_se[industry_map.loc[i, '申万一级行业']]
        except Exception as e:
            industry_map.loc[i, 'max_wei'] = 0.02
    max_wei = industry_map['max_wei'].values

    x = cp.Variable(len(ret_tmp), nonneg=True)

    q = ret_tmp.values
    P = lamda * e_tmp.values

    ind_wei = np.dot(dummies.T, wei_tmp)               # b.shape
    ind_wei_su = pd.Series(ind_wei, index=dummies.columns)
    dum = dummies.T.values                             # A.shape

    para_dict = {'x': x,
                 'max_wei': max_wei,
                 'in_benchmark_wei': in_benchmark_wei,
                 'is_in_bench': is_in_bench,
                 'ret_e': ret_tmp,
                 'dum': dum,
                 'wei_tmp': wei_tmp,
                 'ind_wei': ind_wei,
                 'risk_factor_dict': risk_factor_dict,
                 'limit_factor_df': limit_factor_df,
                 'pre_w': pre_w,
                 'P': P,
                 'total_wei': 1,
                 }
    con_dict = {'in_benchmark': in_benchmark,
                'industry_max_expose': industry_max_expose,
                'turnover': turnover,
                'te': te,
                }

    constraints = generates_constraints(para_dict, con_dict)
    prob = generates_problem(q, x, P, c, pre_w, constraints, te)

    print('开始优化...')
    time_start = time.time()
    prob.solve()
    status = prob.status
    # 如果初始条件无解，需要放松风险因子的约束
    iters = 0
    while status != 'optimal' and iters < 3:
        if len(risk_factor_dict) > 0 and iters == 0:
            tmp_d = deepcopy(risk_factor_dict)
            for k, v in tmp_d.items():
                tmp_d[k] = v + 0.5
            para_dict['risk_factor_dict'] = tmp_d

        elif not turnover and iters == 1:
            turnover = turnover + 0.2
            con_dict['turnover'] = turnover
        elif iters == 2:
            industry_max_expose = industry_max_expose + 0.05
            con_dict['industry_max_expose'] = industry_max_expose

        iters = iters + 1
        constraints = generates_constraints(para_dict, con_dict)
        prob = generates_problem(q, x, P, c, pre_w, constraints, te)
        print('第{}次优化'.format(iters))
        prob.solve()
        status = prob.status

    time_end = time.time()
    print('优化结束，用时', time_end - time_start)
    print('优化结果为{}'.format(status))

    # if prob.status != 'optimal':
    #     input('input:未得出最优解，请检查')
    # np.sum(x.value)
    # np.sum(x.value > 0.0)
    # np.sum(x.value > 0.001)
    # np.sum(x.value[x.value > 0.001])
    # np.sum(x.value[x.value < 0.001])
    # 返回值
    wei_ar = np.array(x.value).flatten()  # wei_ar.size
    wei_se = pd.Series(wei_ar, index=ret_tmp.index)

    # 设定标准，一般情况下无需对股票数量做二次优化，只有股票数量过多是才需要。
    if np.sum(x.value > 0.001) > max_num:
        print('进行第二轮股票数量的优化')
        # wei_selected, n2, tobe_opt = select_import_wei(wei_se, max_num)
        tobe_opt = list(wei_se[wei_se > 0.001].index)
        print('第二次优化为从{}支股票中优化选择出{}支'.format(len(tobe_opt), max_num))

        # 经过处理后，需要优化的计算量大幅度减少。比如第一次优化后，权重大于0.001的股票数量是135，超过最大要求的100。
        # 我们首先保留其中前90，然后从后面的45个中选择10保留下来。
        len(tobe_opt)
        e_tmp2 = e_tmp.loc[tobe_opt, tobe_opt]
        ret_tmp2 = ret_tmp[tobe_opt]
        # wei_tmp2 = wei_tmp[tobe_opt]

        is_in_bench2 = is_in_bench[tobe_opt]

        dummies2 = pd.get_dummies(industry_map.loc[tobe_opt, industry_map.columns[0]])
        dum2 = dummies2.T.values
        # 小坑
        new_ind = ind_wei_su[dummies2.columns]
        new_ind = new_ind / new_ind.sum()
        ind_wei2 = new_ind.values

        # 对个股权重优化的坑，开始时是行业权重乘以0.75，但在二次优化的时候，可能有的行情的权重不够用了。
        max_wei2 = 3 * industry_map.loc[tobe_opt, 'max_wei'].values
        total_wei = 1
        if pre_w:
            pre_w = pre_w[tobe_opt]

        P2 = lamda * e_tmp2.values
        # 有些行业个股权重以前的不够了
        x = cp.Variable(len(ret_tmp2), nonneg=True)
        y = cp.Variable(len(ret_tmp2), boolean=True)
        para_dict2 = {'x': x,
                      'y': y,
                      'y_sum': max_num,       # - n2,
                      'max_wei': max_wei2,  # max_wei2.max()    max_wei2.sum()
                      'in_benchmark_wei': in_benchmark_wei,
                      'is_in_bench': is_in_bench2,
                      'ret_e': ret_tmp2,
                      'dum': dum2,
                      'wei_tmp': wei_tmp,
                      'ind_wei': ind_wei2,    # ind_wei2.sum()
                      'risk_factor_dict': risk_factor_dict,
                      'limit_factor_df': limit_factor_df,
                      'pre_w': pre_w,
                      'P': P,
                      'total_wei': total_wei
                      }
        con_dict2 = {'in_benchmark': in_benchmark,
                     'industry_max_expose': industry_max_expose,
                     'turnover': turnover,
                     'te': te,
                     }
        q2 = ret_tmp2.values
        # P2.shape
        # q2.shape
        # ind_wei2.sum()
        # max_wei2.sum()
        cons = generates_constraints(para_dict2, con_dict2)
        prob = cp.Problem(cp.Maximize(q2.T * x - cp.quad_form(x, P2)), cons)
        prob.solve(solver=cp.ECOS_BB, feastol=1e-10)
        print(prob.status)
        if prob.status != 'optimal':
            input('input:二次股票数量优化时，未得出最优解，请检查')
        # winsound.Beep(600, 2000)
        # print(x.value)
        # print(y.value)
        # np.sum(x.value > 0.001)
        # np.sum(x.value)
        # np.sum(y.value)
        # np.sum(x.value[y.value == 1])


    #
    # prob = cp.Problem(cp.Maximize(q.T * x - cp.quad_form(x, P)),  # - cp.quad_form(x, P)),
    #                   constraints)
    # print(prob.is_dcp())
    # prob.solve()
    # print(prob.status)
    # print(x.value)
    #
    # np.sum(x.value > 0.01)
    #
    # # np.vstack((a, b))  # 在垂直方向上拼接
    # # np.hstack((a, b))  # 在水平方向上拼接
    # industry_max_expose = 0.05

    #
    # if max_num:
    #     '''
    #     优化目标函数：
    #     ECOS is a numerical software for solving convex second-order cone programs (SOCPs) of type
    #     min c'*x
    #     s.t. A * x = b
    #     G * x <= _K h
    #     步骤：
    #     1，假设股票数量没有约束，求解组合优化，得到绝对权重向量
    #     2，对股票数量不过N_max的原始可行域进行限制，选股空间为w1中有权重（>1e-6），数量为n1
    #        强制保留w1中各行业最大权重股以及其他权重靠前的股票，数量为n2，n2<N_max
    #     3，在第2步限制后的可行域内运用BB算法求解最优权重，设置最大迭代刺猬niters，超过
    #        迭代次数返回截至目前的最优解。
    #     '''
    #     # 步骤1
    #     sol = solvers.qp(P, q, G, h, A, b)
    #     wei = sol['x']    # print(wei)  wei.size
    #     wei_ar = np.array(wei).flatten()  # wei_ar.size
    #     n1 = np.sum(wei_ar > 0)
    #     #  np.sum(wei_ar[wei_ar > 0.01])
    #     wei_se = pd.Series(wei_ar, index=ret_tmp.index)
    #     # 步骤2
    #     wei_selected, n2 = select_import_wei(wei_se)
    #     # 步骤3
    #     wei_selected, n2
    #
    #     x = cp.Variable(len(ret_tmp), nonneg=True)
    #     y = cp.Variable(len(ret_tmp), boolean=True)
    #     prob = cp.Problem(cp.Maximize(q.T * x - cp.quad_form(x, P)),  # - cp.quad_form(x, P)),
    #                       [  # G @ x <= h,                              # print(P)   G.size  h.size
    #                           x - y <= 0,
    #                           A @ x == b,
    #                           cp.sum(x) == 1,
    #                           cp.sum(y) <= 200,
    #                       ])
    #     print(prob.is_dcp())
    #     # max_iters: maximum number of iterations
    #     # reltol: relative accuracy(default: 1e-8)
    #     # feastol: tolerance for feasibility conditions (default: 1e-8)
    #     # reltol_inacc: relative accuracy for inaccurate solution (default: 5e-5)
    #     # feastol_inacc: tolerance for feasibility condition for inaccurate solution(default:1e-4)
    #
    #     prob.solve(solver=cp.ECOS_BB, max_iters=20000, feastol=1e-4, reltol=1e-4, reltol_inacc=1e-4,
    #                feastol_inacc=1e-4)
    #     # prob.solve(solver=cp.ECOS_BB,  max_iters=20, feastol=1e-5, reltol=1e-5, feastol_inacc=1e-1)
    #     print(prob.status)
    #     print(x.value)
    #     print(y.value)
    #
    #     max_num
    #
    #
    #     pass
    #
    # # print(cvxpy.installed_solvers())
    # #
    # # np.sum(A, axis=1)
    # # A.size
    # # np.linalg.matrix_rank(A)
    # #
    # # sol = solvers.qp(P, q, G, h, A, b)
    # # wei = sol['x']  # print(wei)  wei.size
    # #
    # # np.rank(A)
    # # print(A)
    # # print(q.T * wei)
    # #
    # # #  np.sum(wei)
    # # wei_ar = np.array(wei).flatten()  # wei_ar.size
    # # #  np.sum(wei_ar > 0.01)
    # # #  np.sum(wei_ar[wei_ar > 0.01])

    return wei_se

