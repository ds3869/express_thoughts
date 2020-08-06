import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from datetime import datetime
import shelve
from talib import abstract
from rqalpha.api import *
from rqalpha import run_func
import tushare as ts
from WindPy import *
from roe_select.reorganize_data import scopy_condition, rise_condition, append_df
from barra_cne6.barra_template import Data
from utility.related_to_rqalpha import change_code_f_rqalpha, select_stocks, add_stock_pool_txt, bt_by_rqalpha
from utility.relate_to_tushare import stocks_basis, generate_months_ends
from canslim.singal_stock_timing import position_control, back_test
from utility.factor_data_preprocess import adjust_months, align
from utility.relate_to_tushare import stocks_basis, trade_days
from barra_cne6.compute_factor_2 import scaler, ma
from utility.macro_data_process import fill_na_by_proceed


factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）'
origin_factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
f_list = os.listdir(factor_path)

date_list = [datetime.strptime(f.split('.')[0], "%Y-%m-%d") for f in f_list]
month_ret = pd.Series(index=date_list)


def from_stock_wei_2_industry_wei(wei_df):
    data = Data()
    all_stocks_code = data.stock_basic_inform
    all_stocks_code = all_stocks_code[['申万一级行业']]
    wei_df = wei_df.fillna(0)
    res = pd.DataFrame()
    for col, se in wei_df.iteritems():
        tmp_df = pd.DataFrame({col: se})
        to_group = pd.concat([tmp_df, all_stocks_code], axis=1)
        to_group.fillna(0, inplace=True)
        grouped = to_group.groupby('申万一级行业').sum()
        if 0 in grouped.index:
            grouped.drop(0, axis=0, inplace=True)
            res = pd.concat([res, grouped], axis=1)

    return res


def name_to_code(names_list):
    data = Data()
    all_stocks_code = data.stock_basic_inform
    all_stocks_code = all_stocks_code[['sec_name'.upper()]]
    all_stocks_code['code'.upper()] = all_stocks_code.index

    all_stocks_code = all_stocks_code.set_index('sec_name'.upper())
    res = list(all_stocks_code.loc[names_list, 'code'.upper()])
    return res


def code_to_name(code_list):
    data = Data()
    all_stocks_code = data.stock_basic_inform
    all_stocks_code = all_stocks_code[['sec_name'.upper()]]
    res = list(all_stocks_code.loc[code_list, 'sec_name'.upper()])
    return res


# 获得特点行业每期股票的数量
def section_stock_num(industry_name):

    data = Data()
    sw_1 = data.industry_sw_1
    stock_codes = list(sw_1.index[sw_1[sw_1.columns[0]] == industry_name])

    res_i = []
    res_v = []

    fn_list = os.listdir(origin_factor_path)
    for i in fn_list:
        dat = pd.read_csv(os.path.join(origin_factor_path, i), encoding='gbk')
        dat.set_index('code', inplace=True)
        new_index = [c for c in stock_codes if c in dat.index]
        dat = dat.loc[new_index, :]

        res_i.append(pd.to_datetime(i.split('.')[0]))
        res_v.append(len(dat))

    res = pd.Series(res_v, index=res_i)

    return res


# t = section_stock_num('计算机')
# c = pd.read_csv(r'D:\pythoncode\IndexEnhancement\股票池\计算机\剔除后股票数量.csv', encoding='gbk')
# c.index = pd.to_datetime(c['date'])
# del c['date']
#
# from matplotlib.ticker import FuncFormatter
#
# np.mean(c['num']/t)
#
# plt.plot(c['num']/t, color='brown', linestyle=':')
#
# def to_percent(temp, position):
#     return format(temp, '.2%')
#
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#
# plt.show()

# 把Bool型转变为数值型
def get_wei(d_df, index_weights=1):
    # d_df.sum()
    d_ar = d_df.values
    tmp_a = np.full(d_ar.shape, 0)
    tmp_a[d_ar==True] = 1

    # tmp_a.sum(axis=0)
    h, l = tmp_a.shape
    res_a = np.full(d_ar.shape, 0.0)
    for i in range(0, l):
        w_s = tmp_a[:, i]/tmp_a[:, i].sum()
        if isinstance(index_weights, (int, float)):
            w_s = w_s * index_weights
        else:
            w_s = w_s*index_weights[i]
        # w_s.sum()
        res_a[:, i] = copy.deepcopy(w_s)

    wei = pd.DataFrame(res_a, index=d_df.index, columns=d_df.columns)

    return wei





# 测试模式，在上面选择出来的股票池中，测试某些因子在该股票池中的分层，来判断该因子的效果。因为是在该股票池中的分层，
# 而不是在全A里面的分层，所以要在选择出来股票池后，再进行，如PEG，资产负责率，ESG等等。
def test_mode(pool, para_dict):
    factor_name = para_dict['因子名称']
    max_para = para_dict['max_para']
    min_para = para_dict['min_para']

    new_pool = pd.DataFrame()

    for col in pool.columns:
        fn = col.strftime("%Y-%m-%d") + '.csv'
        factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'

        data = pd.read_csv(os.path.join(factor_path, fn), engine='python', encoding='gbk')
        if factor_name not in data.columns:
            print('在{}数据中未找到{}因子'.format(fn, factor_name))

        data = data[['code', factor_name]]
        data = data.set_index('code')

        stocks_in_pool = pool[col]
        selected = list(stocks_in_pool[stocks_in_pool == True].index)
        new_index = [i for i in selected if i in data.index]
        data = data.loc[new_index, factor_name]

        data_scalered = scaler(data, 100, 1)
        s1 = data_scalered >= min_para
        s2 = data_scalered <= max_para
        tmp = s1 & s2
        tmp_df = pd.DataFrame({col: tmp})

        new_pool = pd.concat([new_pool, tmp_df], axis=1)

    new_pool.fillna(False, inplace=True)

    return new_pool


def layer_test(factor_name, pool):

    para_dict = {'因子名称': factor_name,
                 'max_para': 30,
                 'min_para': 1,
                }

    pool_tmp = test_mode(pool, para_dict)
    pool_tmp = append_df(pool_tmp, target_feq='D')
    pool_wei_tmp = get_wei(pool_tmp)
    pool_wei_tmp = append_df(pool_wei_tmp, target_feq='D')

    # 简单回测
    daily_return_bottom, net_cpd_1_30, cum_excess_df = easy_bt(pool_wei_tmp, basic_return_path=None)

    para_dict = {'因子名称': factor_name,
                 'max_para': 69,
                 'min_para': 31,
                 }

    pool_tmp = test_mode(pool, para_dict)
    pool_tmp = append_df(pool_tmp, target_feq='D')
    pool_wei_tmp = get_wei(pool_tmp)
    pool_wei_tmp = append_df(pool_wei_tmp, target_feq='D')

    # 简单回测
    daily_return, net_cpd_31_69, cum_excess_df = easy_bt(pool_wei_tmp, basic_return_path=None)

    para_dict = {'因子名称': factor_name,
                 'max_para': 100,
                 'min_para': 70,
                 }

    pool_tmp = test_mode(pool, para_dict)
    pool_tmp = append_df(pool_tmp, target_feq='D')
    pool_wei_tmp = get_wei(pool_tmp)
    pool_wei_tmp = append_df(pool_wei_tmp, target_feq='D')

    # 简单回测
    daily_return_top, net_cpd_70_100, cum_excess_df = easy_bt(pool_wei_tmp, basic_return_path=None)

    res_df = pd.DataFrame({'top': net_cpd_70_100['net_value'], 'middle': net_cpd_31_69['net_value'],
                           'bottom': net_cpd_1_30['net_value']})

    fig = plt.figure()
    plt.plot(res_df)
    plt.legend(list(res_df.columns))
    plt.show()

    top_sub_bottom = daily_return_top['daily_return'] - daily_return_bottom['daily_return']
    top_sub_bottom_cum = (top_sub_bottom + 1).cumprod()
    top_sub_bottom_df = pd.DataFrame({"top减bottom累计收益": top_sub_bottom_cum})
    top_sub_bottom_df.index.name = '日期'

    fig = plt.figure()
    plt.plot(top_sub_bottom_df)
    plt.show()

    save_path = r'D:\pythoncode\IndexEnhancement\股票池\计算机'
    res_df.to_csv(os.path.join(save_path, factor_name+'_3.csv'))
    top_sub_bottom_df.to_csv(os.path.join(save_path, factor_name+'_top减bottom_3.csv'))


def get_stock_pool(financial_condition_dict, factor_condition_dict, factor_range, indus_dict, start_date=None,
                   end_date=None):

    if financial_condition_dict:
        # 财务绝对要求条件的股票池构建，如roettm大于5%，eps同比增速大于5，sue位于所有股票的top5%。
        stock_pool_financial = financial_condition_pool(financial_condition_dict, start_date=None, end_date=None)
    else:
        stock_pool_financial = None

    stock_pool_financial.sum()
    if factor_condition_dict:
        # 因子化选股条件，如经过行业中性化处理后的SUE因子在top 5%
        stock_pool_factor = factor_condition_pool(factor_condition_dict, factor_range, indus_dict, start_date=None, end_date=None)
    else:
        stock_pool_factor = None

    stock_pool = concat_stock_pool(stock_pool_financial, stock_pool_factor)

    # indus_dict = {'to_handle_indus': ['银行', '非银金融', '有色金属', '化工', '钢铁', '采掘', '国防军工'],
    #               # keep 和 delete 两种模式， keep模型下，保留下来上个变量定义的子行业，delete模式下，删除上面的子行业
    #               'handle_type': 'delete',
    #              }

    # tmp = stock_pool.sum()

    if indus_dict['handle_type'] == 'delete':
        stock_pool = del_industry(stock_pool, indus_dict['to_handle_indus'])
    elif indus_dict['handle_type'] == 'keep':
        stock_pool = keep_industry(stock_pool, indus_dict['to_handle_indus'])

    # 选择给定时间段的列
    cols = [col for col in stock_pool.columns if end_date >= col >= start_date]
    stock_pool = stock_pool[cols]

    return stock_pool


# 对stock_pool中属于to_del_indus的行业，把True换成False
def del_industry(stock_pool, to_del_indus):
    data = Data()
    stock_basic = data.stock_basic_inform
    sw_1 = stock_basic[['申万一级行业']]

    for col, v in stock_pool.iteritems():
        for ind in to_del_indus:
            si = [i for i in sw_1[sw_1[sw_1.columns[0]] == ind].index if i in stock_pool.index]
            v[si] = False

    return stock_pool


def read_icir_e(icir_path):
    #icir_path = r'D:\pythoncode\IndexEnhancement\单因子检验\icir.csv'
    icir_e = pd.read_csv(icir_path, encoding='gbk')
    icir_e.set_index('date', inplace=True)
    icir_e.index = pd.to_datetime(icir_e.index)
    icir_e = icir_e.shift(1, axis=0)
    icir_e.dropna(how='all', axis=0, inplace=True)


def keep_market():

    pass


# 对stock_pool中不属于to_keep_indus的行业，把True换成False
def keep_industry(stock_pool, to_keep_indus_list, industry_type='sw'):
    data = Data()

    stock_basic = data.stock_basic_inform

    if industry_type == 'sw':
        industry_df = stock_basic[['申万一级行业']]
    elif industry_type == 'zx':
        industry_df = stock_basic[['中信一级行业']]

    for col, v in stock_pool.iteritems():
        for ind in to_keep_indus_list:
            # 选出该行业的股票
            tmp = industry_df[industry_df[industry_df.columns[0]] == ind].index
            # 也在stock_pool的index里面
            si = [i for i in tmp if i in stock_pool.index]
            # 其余行业的股票
            se = list(set(stock_pool.index).difference(set(si)))
            # 均为否
            v[se] = False

    return stock_pool


# 对stock_pool中的股票，如果满足条件的股票较多，根据成长和盈利因子综合打分，选择综合排名靠前的股票。
def select_stocks_by_scores(stock_pool, factors, factor_weight, reversed_factors, icir_e, max_num=100,
                            each_industry_num=5, select_type='by_industry', wei_type='equal'):

    data = Data()
    industry_sw = data.industry_sw_1

    new_stock_pool = pd.DataFrame()
    for col, value in stock_pool.iteritems():
        # if value.sum() > max_num:
        codes = list(value[value == True].index)
        codes_selected = select_stocks_by_scores_singal_section(codes, col, factors, factor_weight, reversed_factors,
                                                                industry_sw, max_num, icir_e, each_industry_num,
                                                                select_type=select_type,  wei_type=wei_type)
        # else:
        #     codes_selected = list(value[value==True].index)

        if codes_selected:
            tmp = pd.DataFrame(np.full(len(codes_selected), True), index=codes_selected, columns=[col])
            new_stock_pool = pd.concat([new_stock_pool, tmp], axis=1)

    new_stock_pool.fillna(False, inplace=True)

    return new_stock_pool


def select_stocks_by_scores_singal_section(codes, dt, factors_dict, factor_weight, reversed_factors, industry_infor,
                                           max_num, icir_e, each_industry_num, select_type='by_industry',
                                           wei_type='equal', industry_selected=None):

    # 得到ICIR值
    if wei_type != 'equal' and dt not in icir_e.index:
        return None

    expect_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\预期与概念数据'
    expect_f_list = os.listdir(expect_path)

    # 读取dt期的因子数据，并预处理
    # factor_path = r'D:\pythoncode\IndexEnhancement\RPS\因子（已预处理）'
    factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    f_list = os.listdir(factor_path)
    fn = dt.strftime("%Y-%m-%d") + '.csv'
    if fn not in f_list:
        # print('未找到'+ fn + '该期数据')
        return None
    data = pd.read_csv(os.path.join(factor_path, fn), engine='python', encoding='gbk')
    data = data.set_index('code')
    new_index = [c for c in codes if c in data.index]
    data = data.loc[new_index, :]

    # if fn in expect_f_list:
    #     # 仅用来剔除无分析师覆盖得股票，未充分使用一致预期数据
    #     expect_data = pd.read_csv(os.path.join(expect_path, fn), engine='python', encoding='gbk')
    #     expect_data = expect_data.set_index('code')
    #     # expect_df = expect_data[['WEST_AVGROE_YOY', 'WEST_NETPROFIT_YOY', 'WEST_SALES_YOY']]
    #     data = pd.concat([data, expect_data], axis=1, join='inner')
    #     # new_index = [c for c in data.index if c in expect_data.index]
    #     # data = data.loc[new_index, :]

    # 根据因子字典，分别计算得分
    scores_total = pd.DataFrame()
    for key, v_list in factors_dict.items():
        dat_tmp = data[v_list]
        scores_tmp = get_scores(dat_tmp, reversed_factors)
        scores_df = pd.DataFrame({key: scores_tmp})
        scores_total = pd.concat([scores_total, scores_df], axis=1)

    if wei_type == 'icir':  # icir加权
        icir_section = icir_e.loc[dt, :]

        tmp_se = pd.Series(index=list(factors_dict.keys()))
        for key, value in factors_dict.items():
            tmp_se[key] = np.nanmean(icir_section[value])
            if key in reversed_factors:
                tmp_se[key] = -1*tmp_se[key]

        wei_se = pd.Series(index=list(factors_dict.keys()))
        # 通过icir得到不同因子之间的权重
        if tmp_se['value'] < 0 and tmp_se['growth'] > 0:
            wei_se['value'] = 0
            wei_se['growth'] = 1
        elif tmp_se['value'] > 0 and tmp_se['growth'] < 0:
            wei_se['growth'] = 0
            wei_se['value'] = 1
        elif tmp_se['value'] < 0 and tmp_se['growth'] < 0:
            wei_se['growth'] = 0.5
            wei_se['value'] = 0.5
        else:
            wei_se['growth'] = tmp_se['value']/(tmp_se['value'] + tmp_se['growth'])
            wei_se['value'] = tmp_se['growth']/(tmp_se['value'] + tmp_se['growth'])

        tmp = [i for i in wei_se.index if i not in ['growth', 'value']]
        wei_se[tmp] = tmp_se[tmp]/np.nansum(tmp_se[tmp])

        for i in wei_se.index:
            for v in factor_weight.values():
                if i in v[0]:
                    wei_se[i] = wei_se[i] * v[1]
                    continue

        # 确保排序一样
        scores_total = scores_total[wei_se.index]

        va = np.dot(scores_total.values, wei_se.values)
        scores_sum = pd.Series(va, index=scores_total.index)

    # 加权方式
    if wei_type == 'equal':      # 等权合成
        scores_sum = scores_total.sum(axis=1)

    # 排序
    scores_sorted = scores_sum.sort_values(ascending=False)

    # 选股方式
    if select_type == 'by_industry':
        scores_sorted = pd.concat([pd.DataFrame({'scores': scores_sorted}), industry_infor], axis=1, join='inner')

        res = pd.DataFrame()
        grouped = scores_sorted.groupby('申万一级行业')
        for key, value in grouped:
            if len(value) > each_industry_num:
                tmp = value.iloc[:each_industry_num, :]
                res = pd.concat([res, tmp], axis=0)
            else:
                res = pd.concat([res, value], axis=0)

        res = list(res.index)
    elif select_type == 'total_num':
        res = list(scores_sorted.index[:max_num])

    # 从特定行业选择排名靠前的股票
    elif select_type == 'special_industry':
        if not industry_selected:
            print('无行业信息，bug')

        scores_sorted = pd.concat([pd.DataFrame({'scores': scores_sorted}), industry_infor], axis=1, join='inner')

        res = pd.DataFrame()
        grouped = scores_sorted.groupby(industry_infor.columns[0])
        for key, value in grouped:
            if key in industry_selected:
                if len(value) > each_industry_num:
                    tmp = value.iloc[:each_industry_num, :]
                    res = pd.concat([res, tmp], axis=0)
                else:
                    res = pd.concat([res, value], axis=0)

        res = list(res.index)

    return res


# 计算成长性得分 : 去极值、标准化、求和、归一化为1到100
def get_scores(dat, reversed_factors=[]):

    # dat.applymap(lambda x: float(x) if isinstance(x, str) else x)

    new_dat = pd.DataFrame()
    for col, v in dat.iteritems():

        v = pd.to_numeric(v, errors='coerce')
        vals = v.values

        # 去极值
        dm = np.nanmedian(vals)
        dm1 = np.nanmedian(np.abs(vals - dm))
        vals = np.where(vals > dm + 5 * dm1, dm + 5 * dm1, np.where(vals < dm - 5 * dm1, dm - 5 * dm1, vals))
        # 标准化
        if col in reversed_factors:
            vals = (np.nanmean(vals) - vals) / np.nanstd(vals)
        else:
            vals = (vals - np.nanmean(vals)) / np.nanstd(vals)
        tmp = pd.DataFrame(vals, index=dat.index, columns=[col])
        tmp.fillna(0, inplace=True)

        new_dat = pd.concat([new_dat, tmp], axis=1)

    scores = new_dat.sum(axis=1)
    # 归一化为1到100
    # scores = scaler(tmp, 100, 1)
    return scores


# 对dat_df中的数据，按照way中列的方式进行两步排序选择，即从第一步选择出来的数据中再按照第二步的要求再次排序。
def twice_sort(dat_df, way):
    [factor_name, type, num] = way['first']
    if type == 'up':
        dat_df = dat_df.sort_values(factor_name, ascending=False)
        dat_df_1 = dat_df.loc[dat_df.index[:num], :]
    elif type == 'down':
        dat_df = dat_df.sort_values(factor_name, ascending=True)
        dat_df_1 = dat_df.loc[dat_df.index[:num], :]

    [factor_name_2, type_2, num_2] = way['second']
    if type_2 == 'up':
        dat_df_1 = dat_df_1.sort_values(factor_name_2, ascending=False)
        dat_df_2 = dat_df_1.loc[dat_df_1.index[:num_2], :]
    elif type_2 == 'down':
        dat_df_1 = dat_df_1.sort_values(factor_name_2, ascending=True)
        dat_df_2 = dat_df_1.loc[dat_df_1.index[:num_2], :]

    return dat_df_2


def rps_factor(rps_min=50, rps_max=100):
    data = Data()
    rps = data.RPS
    rps.fillna(0, inplace=True)

    rps_cond = float_2_bool_df(rps, min_para=rps_min, max_para=rps_max)
    # 用于回测,向右移动一期
    rps = rps.shift(1, axis=1)
    rps_cond = rps_cond.shift(1, axis=1)
    rps.dropna(axis=1, how='all', inplace=True)
    rps_cond.dropna(axis=1, how='all', inplace=True)

    return rps_cond


def easy_bt(wei_stocks, basic_return_infor):
    data = Data()
    changepct_daily = data.CHANGEPECT_OPEN_DAILY
    changepct_daily = changepct_daily.shift(-1, axis=1)
    changepct_daily.dropna(how='all', axis=1, inplace=True)

    changepct_daily = changepct_daily / 100

    wei_stocks, changepct_daily = align(wei_stocks, changepct_daily)

    # fee_type='No_fee' 不计算佣金和印花税， 'fee_1'计算佣金和印花税，不计算冲击成本
    daily_return, net_value = back_test(changepct_daily, wei_stocks, fee_type='fee_1')
    # plt.plot(net_cpd)

    # 若有基准日度收益率，则计算累计超额收益率
    if isinstance(basic_return_infor, str):
        # 有基准收益，算超额收益
        basic_return = pd.read_csv(basic_return_infor, engine='python')
        basic_return = basic_return.set_index('date')
        if 'daily_return' in basic_return.columns:
            daily_excess_r = daily_return['daily_return'] - basic_return['daily_return']
        # 若没有日度收益数据，则根据日度净值数据计算出日度收益收益数据
        elif 'daily_return' not in basic_return.columns and 'net_value' in basic_return.columns:
            basic_return['daily_return'] = basic_return['net_value']/basic_return['net_value'].shift(1) - 1
            daily_excess_r = daily_return['daily_return'] - basic_return['daily_return']
            daily_excess_r.dropna(inplace=True)

        daily_excess_cum = (daily_excess_r + 1).cumprod()
        cum_excess_df = pd.DataFrame({'cum_excess_ret': daily_excess_cum})

    elif isinstance(basic_return_infor, pd.DataFrame):
        if 'daily_return' not in basic_return_infor.columns and 'net_value' in basic_return_infor.columns:
            basic_return_infor['daily_return'] = basic_return_infor['net_value'] / \
                                                 basic_return_infor['net_value'].shift(1) - 1
            daily_excess_r = daily_return['daily_return'] - basic_return_infor['daily_return']
            daily_excess_r.dropna(inplace=True)

        daily_excess_cum = (daily_excess_r + 1).cumprod()
        cum_excess_df = pd.DataFrame({'cum_excess_ret': daily_excess_cum})
    else:
        cum_excess_df = None

    return daily_return, net_value, cum_excess_df


# 把月度内的日度True和False，拓展为月度
def cond_append_to_month(d_df):

    month_ends = generate_months_ends()
    # 在d_df里的月末日期
    me_in_d = [me for me in month_ends if me in d_df.columns]
    #
    m_e_loc_list = [np.where(d_df.columns == me)[0][0] for me in me_in_d]
    m_s_loc_list = [me + 1 for me in m_e_loc_list]
    # 表头插入
    m_s_loc_list = [0] + m_s_loc_list
    # 删除最后一个
    m_s_loc_list.pop()
    # len(m_e_loc_list)
    # len(m_s_loc_list)

    res_df = pd.DataFrame()
    pre_se = pd.Series(True, index=d_df.index)

    for col, se in d_df.iteritems():
        loc = np.where(d_df.columns == col)[0][0]
        # 月初第一个交易日
        if loc in m_s_loc_list:
            pre_se = pd.Series(True, index=d_df.index)

        res_se = pre_se & se
        res_df = pd.concat([res_df, pd.DataFrame({col: res_se})], axis=1)
        pre_se = res_se

    return res_df


# 把 dat_df中大于等于min_para, 或是小于等于max_para的值变成True，其他值变成False
def float_2_bool_df(dat_df, min_para=None, max_para=None):

    dat_array = dat_df.values
    # res_a = np.full(dat_array.shape, False)

    if min_para:
        min_c = dat_array >= min_para
        res_a = min_c
    if max_para:
        max_c = dat_array <= max_para
        res_a = max_c

    if min_para and max_para:
        res_a = np.logical_and(min_c, max_c)

    res = pd.DataFrame(data=res_a, index=dat_df.index, columns=dat_df.columns)

    return res


def save_each_sec_name(dat_df, save_path=None, save_name=None):
    basis = stocks_basis()
    basis = basis.set_index('ts_code')

    if not save_path:
        save_path = r'D:\pythoncode\IndexEnhancement\股票池'
    if not save_name:
        save_name = '往期股票池'

    maxl = np.sum(dat_df == True, axis=0).max()
    res_to_csv = pd.DataFrame(index=range(0, maxl))

    for col, items in dat_df.iteritems():
        selected = items[items.index[items == True]]
        selected = [i for i in selected.index]
        if len(selected) > 0:
            s_name = basis.loc[selected, 'name']
            s_name = pd.DataFrame(s_name.values, index=range(0, len(s_name)), columns=[col])

            res_to_csv = pd.concat([res_to_csv, s_name], axis=1)

    res_to_csv.to_csv(os.path.join(save_path, save_name+'.csv'), encoding='gbk')


# 财务数据选股部分
def financial_condition_pool(selection_dict, start_date, end_date):
    data = Data()
    stock_basic = data.stock_basic_inform
    firstindustry = stock_basic[['中信一级行业']]

    all_stocks_code = stock_basic[['sec_name'.upper(), 'ipo_date'.upper()]]

    # roe
    roettm = data.roettm
    # 净利润同比增速
    netprofitgrowrate = data.netprofitgrowrate
    # 基本每股收益同比增长率
    basicepsyoy = data.basicepsyoy
    # 销售毛利率
    grossincome = data.grossincomeratiottm
    # 资产负债率
    debtassetsratio = data.debtassetsratio
    # 估值
    pe = data.pe

    cond_total = pd.DataFrame()
    for plate_name, conditions_dict in selection_dict.items():
        pe_con = None
        con_plate = None
        if plate_name == 'all':
            codes_in_industry = list(firstindustry.index)
        else:
            codes_in_industry = [ind for ind in firstindustry.index if firstindustry[ind] == plate_name]

        for conditions_type, tuples in conditions_dict.items():
            if conditions_type.split('_')[0] == 'scope':
                myfun = 'scopy_condition'
                for t in tuples:
                    if t == 'pe':
                        has_pe = 1
                        pe_con = copy.deepcopy(t)
                        continue

                res = eval('select_stocks(' + tuples[0] + ', codes_in_industry, start_date, end_date)')
                res_con = eval(myfun + '(res, minV=' + str(tuples[1]) + ', maxV= ' + str(tuples[2]) + ')')

                if isinstance(con_plate, pd.DataFrame):
                    con_plate = con_plate & res_con
                else:
                    con_plate = res_con

            elif conditions_type.split('_')[0] == 'rise':
                myfun = 'rise_condition'
                # for t in tuples:

                res = eval('select_stocks(' + tuples[0] + ', codes_in_industry, start_date, end_date)')
                res_con = eval(myfun + '(res,' + str(tuples[1]) + ')')

                if isinstance(con_plate, pd.DataFrame):
                    con_plate = con_plate & res_con
                else:
                    con_plate = res_con

        # 不同行业之间合并
        cond_total = pd.concat([cond_total, con_plate], axis=1)

    # 剔除上市未满N年得股票，N = 1
    # N = 1
    # for col, items in cond_total.iteritems():
    #     for i in items.index:
    #         if items[i]:
    #             if i in all_stocks_code.index:
    #                 de = col - all_stocks_code.loc[i, 'ipo_date']
    #                 if de.days < N * 365:
    #                     items[i] = False
    #             else:
    #                 # i 不在all_stocks_code里面，是因为all_stocks_code没有更新，说明该股票是最近上市的股票，
    #                 # 直接全部复制为False
    #                 items[i] = False

    # 剔除st
    for col, items in cond_total.iteritems():
        for i in items.index:
            if i in all_stocks_code.index and 'ST' in all_stocks_code.loc[i, 'sec_name'.upper()]:
                items[i] = False

    # 调整为公告日期
    cond_total = adjust_months(cond_total)
    # 用来扩展月度数据
    cond_total = append_df(cond_total)

    return cond_total


def factor_condition_pool(selection_dict, factor_range, indus_dict, start_date=None, end_date=None):
    # 使用多因子选股的方式
    '''
    if use_type == 'MMF':
        path_dict = {'save_path': r'D:\pythoncode\IndexEnhancement\多因子选股',
                     'factor_panel_path': r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）',
                     'ic_path': r'D:\pythoncode\IndexEnhancement\单因子检验\ic.csv',
                     'old_matrix_path': r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'}

        just_forecast = True

        factors_to_concat = {
            'mom': ['M_reverse_180', 'M_reverse_20', 'M_reverse_60'],
            'liq_barra': ['STOA_Barra', 'STOM_Barra', 'STOQ_Barra'],
            'vol': ['std_1m', 'std_3m', 'std_6m', 'std_12m'],
            'value': ['EP', 'BP'],
            'quality': ['ROE_q'],
            'growth': ['SUE', 'REVSU'],
            # 'size': ['nonlinearsize_barra'],
        }

        est_stock_rets = multi_factor_model(factors_to_concat=factors_to_concat, path_dict=path_dict,
                                            just_forecast=just_forecast)

        bestN_stock = 200
        industry_demand = None
        stock_wei_equalwei = select_stock_ToF(est_stock_rets, bestN_stock, industry_demand)

        # new_record_stock(stock_wei_equalwei, save_name='每期多因子', save_path=path_dict['save_path'])

        res_pool = stock_wei_equalwei
    # 使用因子分层排序或者top value的方式
    elif use_type == 'layer_way':
    '''
    res_pool = None

    for key, value in selection_dict.items():
        factor_name = key
        top_or_bottom = value[0]
        per = value[1]
        factor_pool = signal_factor_pool(factor_range, indus_dict, factor_name, top_or_bottom, per)

        res_pool = concat_stock_pool(res_pool, factor_pool)

    return res_pool


# 对单个因子的分层选择
def signal_factor_pool(factor_range, indus_dict, factor_name, top_or_bottom, per):
    # factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）'

    if factor_range == 'one_industry':
        data = Data()
        sw_1 = data.industry_sw_1

    factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    f_list = os.listdir(factor_path)

    res_df = pd.DataFrame()
    for f in f_list:
        data = pd.read_csv(os.path.join(factor_path, f), engine='python', encoding='gbk')
        if factor_name not in data.columns:
            print('在{}数据中未找到{}因子'.format(f, factor_name))

        data = data[['code', 'name', factor_name]]
        data = data.set_index('code')
        data.dropna(axis=0, how='any', inplace=True)

        if factor_range == 'one_industry':
            se = list(sw_1.index[sw_1[sw_1.columns[0]] == indus_dict['to_handle_indus'][0]])
            se_code = [i for i in se if i in data.index]
            data = data.loc[se_code, :]

        le = int(len(data) * per)
        sorted_df = data.sort_values(by=factor_name, ascending=False)

        if top_or_bottom == 'top':
            r = sorted_df.index[:le - 1]
        elif top_or_bottom == 'bottom':
            r = sorted_df.index[-le - 1:]

        tmp_df = pd.DataFrame([True for i in range(0, len(r))], index=r.values,
                              columns=[datetime.strptime(f.split('.')[0], "%Y-%m-%d")])
        res_df = pd.concat([res_df, tmp_df], axis=1)
        res_df.fillna(False, inplace=True)

    return res_df


def concat_stock_pool(pool_1, pool_2):

    if not isinstance(pool_1, pd.DataFrame):
        return pool_2

    if not isinstance(pool_2, pd.DataFrame):
        return pool_1

    cols = pool_1.columns.intersection(pool_2.columns)
    inds = pool_1.index.intersection(pool_2.index)

    pool_total = pool_1.loc[inds, cols] & pool_2.loc[inds, cols]
    # pool_total.sum()

    return pool_total


def get_index(industry_name):
    # industry_name = '计算机'
    path = r'D:\pythoncode\IndexEnhancement\barra_cne6\factor_data'
    dat = pd.read_csv(os.path.join(path, 'industry_close.csv'), encoding='gbk')
    dat.index = pd.to_datetime(dat['日期'])
    del dat['日期']

    map_dict = {'农林牧渔': '801010.SI',
                '计算机': '801750.SI'
                }
    indus_code = map_dict[industry_name]

    res = pd.DataFrame({industry_name: dat[indus_code]})

    return res


# 处理宏观数据，把数据转化为方向数据
def form_direction(dat_df, window_dict={}, retained=3):
    ret = pd.DataFrame()

    # 添加Nan, 因为要做均值化处理，有一个有nan的话，连续几个都是nan了。
    # 直接使用前值填充
    dat_df_filled = fill_na_by_proceed(dat_df)

    for col, se in dat_df_filled.iteritems():
        # 根据window_dict里面的内容得到该col的window数据
        finded = retained
        for k, v in window_dict.items():
            if col in v:
                finded = k
        # 计算mean
        tmp_df = pd.DataFrame({col: se})
        ma_df = ma(tmp_df, finded)
        # 删除nan
        ma_df = ma_df.dropna()
        # 得到方向变量
        res = pd.DataFrame(index=ma_df.index, columns=[col])
        for i in range(0, len(ma_df)):
            if i == 0:
                res.iloc[0, 0] = 0
            else:
                if ma_df.iloc[i, 0] > ma_df.iloc[i-1, 0]:
                    res.iloc[i, 0] = 1
                elif ma_df.iloc[i, 0] < ma_df.iloc[i-1, 0]:
                    res.iloc[i, 0] = -1
                else:
                    res.iloc[i, 0] = res.iloc[i-1, 0]

        ret = pd.concat([ret, res], axis=1)

    return ret


# 股票池月度表现分析函数
def month_return_compare_to_market_index(stock_list, his_month):
    data = Data()
    changePCT = data.changepct_monthly

    ff = None
    for c in changePCT.columns:
        if c.year == his_month.year and c.month == his_month.month:
            ff = c
            break
    res2 = changePCT.loc[stock_list, ff]

    index_path = r'D:\pythoncode\IndexEnhancement\指数相关'
    index_price = pd.read_csv(os.path.join(index_path, 'index_price_monthly.csv'), engine='python')
    index_price = index_price.set_index(index_price.columns[0])
    index_price.index = pd.to_datetime(index_price.index)
    index_r = (index_price - index_price.shift(1)) / index_price.shift(1)

    fff = None
    for c in index_r.index:
        if c.year == his_month.year and c.month == his_month.month:
            fff = c
            break

    res1 = pd.DataFrame({fff: index_r.loc[fff, :].drop_duplicates()})

    tt = pd.DataFrame(data=res2.mean(), index=['组合'], columns=[ff])
    res1 = pd.concat([res1, tt], axis=0)

    return res1, res2



