# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:47:11 2019

@author: admin
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from itertools import chain
from functools import reduce
from sklearn.linear_model import LinearRegression
from utility.constant import info_cols
from utility.relate_to_tushare import trade_days


def align(df1, df2, *dfs):
    # chain 是把多个迭代器合成一个迭代器
    dfs_all = [df for df in chain([df1, df2], dfs)]
    # 看df1和df2是否有单个列的
    if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
        dims = 1
    else:
        dims = 2
    # 对日期求交期. reduce: 用传给reduce中的函数function（有两个参数）先对集合中的第 1、2个元素进行操作，
    # 得到的结果再与第三个数据用function函数运算，最后得到一个结果。
    mut_date_range = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
    # 对columns求交集
    mut_codes = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
    # 如果df1和df2都是多维的，求日期和代码的交集；否则，只求日期的交集
    if dims == 2:
        dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
    elif dims == 1:
        dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
    return dfs_all


def get_factor_data(datdf):
    """
    根据输入的因子名称将原始因子截面数据分割
    """
    global info_cols

    cond = pd.Series(True, index=datdf.index)
    for col in ['Is_open1', 'Mkt_cap_float']:
        if pd.isnull(datdf[col]).all():
            continue
        else:
            cond &= ~pd.isnull(datdf[col])

    datdf = datdf.loc[cond]

    return datdf, pd.DataFrame()

    # # 对截面数据中，基准信息列存在空缺的股票（行）进行删除处理
    # # 选出市值因子不为空的
    # tmp_info_cols = [inf for inf in info_cols if inf in datdf.columns]
    # try:
    #     cond = ~pd.isnull(datdf['Mkt_cap_float'])
    # except Exception as e:
    #     print('debug')
    #
    # datdf = datdf.loc[cond]
    #
    # # 逻辑问题，不能因为一些因子就把这个股票给删掉
    # # ['Mkt_cap_float', 'Is_open1']
    # for col in tmp_info_cols:
    #     if col in ['Is_open1', 'Mkt_cap_float', 'Pct_chg_nm']:  # todo 因子命名问题，后续统一一下
    #         # 是否全为 空值， 最后一个月
    #         if pd.isnull(datdf[col]).all():
    #             continue
    #     # 逐列 非空 判断， 并逐列求 逻辑and 运算，确保datfd中 info_cols 里面的列名都在。
    #     if col != 'Mkt_cap_float':
    #         try:
    #             # print(col)
    #             cond &= ~pd.isnull(datdf[col])
    #         except Exception as e:
    #             print('debug')
    # datdf = datdf.loc[cond]

    # 将原截面数据按照预处理与否分别划分，返回需处理和不需处理2个因子截面数据
    # if names is None:
    #     return datdf, pd.DataFrame()
    # else:
    #     dat_to_process = datdf.iloc[:, idx]
    #     dat_to_process = pd.merge(datdf[tmp_info_cols], dat_to_process,
    #                               left_index=True, right_index=True)
    #     # 使用set创建两个不重复元素集，想减，得到剩余的元素集合，然后通过sorted转为列。
    #     unchanged_cols = sorted(set(datdf.columns) - set(dat_to_process.columns))
    #     dat_unchanged = datdf[unchanged_cols]
    # return dat_to_process, dat_unchanged


def fill_na(data, ind='sw', fill_type='any'):
    """
    缺失值填充：缺失值少于10%的情况下使用行业中位数代替
    """
    global info_cols
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['Industry_sw']), :]

    tmp_info_cols = [inf for inf in info_cols if inf in datdf.columns]
    # datdf中剔除info_cols后的列名
    facs_to_fill = datdf.columns.difference(set(tmp_info_cols))

    datdf[facs_to_fill] = datdf[facs_to_fill].applymap(coerce_numeric)
    datdf = datdf.replace([np.inf, -np.inf], np.nan)    # 替换inf

    # pd.to_numeric( datdf[facs_to_fill], errors='coerce')
    if fill_type != 'any':
        facs_to_fill = [fac for fac in facs_to_fill            # 筛选缺失值少于10%的因子
                            if pd.isnull(datdf[fac]).sum() / len(datdf) <= 0.1]
    else:
        facs_to_fill = [fac for fac in facs_to_fill            # 筛选缺失值少于10%的因子
                        if pd.isnull(datdf[fac]).any()]

    if ind in ['zx', 'sw']:
        grouped_column = f'Industry_{ind}'
    elif ind == 'Second_industry':
        grouped_column = 'Second_industry'
    else:
        raise Exception

    for fac in facs_to_fill:
        fac_median_by_ind = datdf[[grouped_column, fac]].groupby(grouped_column).median()
        # 把dateframe转为dict,并取fac为key以解决 dict套dict 的问题
        fac_ind_map = fac_median_by_ind.to_dict()[fac]
        # 选出需要替换的数据
        fac_to_fill = datdf.loc[pd.isnull(datdf[fac]), [grouped_column, fac]]
        # map函数可以接受含有映射关系的字典。使用map做行业到其均值的映射。
        fac_to_fill.loc[:, fac] = fac_to_fill[grouped_column].map(fac_ind_map)
        # 添加回到datdf
        datdf.loc[fac_to_fill.index, fac] = fac_to_fill[fac].values
        if pd.isnull(datdf[fac]).any():
            datdf[fac] = datdf[fac].fillna(np.nanmean(datdf[fac]))

    # 针对sw行业存在缺失值的情况
    if len(datdf) < len(data):
        idx_to_append = data.index.difference(datdf.index)
        datdf = pd.concat([datdf, data.loc[idx_to_append, :]])
        datdf.sort_index()

    return datdf


def coerce_numeric(s):
    try:
        return float(s)
    except:
        return np.nan


def winsorize(data, n=5):
    """
    去极值：5倍中位数标准差法（5mad）
    """
    global info_cols
    
    datdf = data.copy()
    tmp_info_cols = [inf for inf in info_cols if inf in datdf.columns]

    # 找出含有 nan 的列
    if_contain_na = pd.isnull(datdf).sum().sort_values(ascending=True)
    facs_to_remove = if_contain_na.loc[if_contain_na > 0].index.tolist()
    if 'PCT_CHG_NM' in facs_to_remove:
        facs_to_remove.remove('PCT_CHG_NM')

    # 剔除含有 nan 的列 和 info_cols的列 后的所有列
    facs_to_win = datdf.columns.difference(set(tmp_info_cols)).difference(set(tuple(facs_to_remove)))
    dat_win = datdf[facs_to_win]
    dat_win = dat_win.applymap(apply_func2)
    fac_vals = dat_win.values

    # np.median(fac_vals)
    try:
        dm = np.nanmedian(fac_vals, axis=0)
    except Exception as e:
        print('debug')
    # 与均值差的绝对值的非 nan 均值
    dm1 = np.nanmedian(np.abs(fac_vals - dm), axis=0)
    if 0 in (dm + n*dm1): 
        # 针对存在去极值后均变为零的特殊情况（2009-05-27-'DP')
        cut_points = [i for i in np.argwhere(dm1 == 0)[0]]
        # 提取对应列，对其不进行去极值处理
        facs_unchanged = [facs_to_win[cut_points[i]] for i in range(len(cut_points))] 
        # 仅对剩余列进行去极值处理
        facs_to_win_median = facs_to_win.difference(set(tuple(facs_unchanged)))
        
        dat_win_median = datdf[facs_to_win_median]

        def fun1(x):
            try:
                r = float(x)
            except Exception as e:
                r = 0
            return r
        dat_win_median = dat_win_median.applymap(fun1)

        fac_median_vals = dat_win_median.values
        dmed = np.nanmedian(fac_median_vals, axis=0)
        dmed1 = np.nanmedian(np.abs(fac_median_vals - dmed), axis=0)
        dmed = np.repeat(dmed.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        dmed1 = np.repeat(dmed1.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        
        fac_median_vals = np.where(fac_median_vals > dmed + n*dmed1, dmed+n*dmed1, 
              np.where(fac_median_vals < dmed - n*dmed1, dmed - n*dmed1, fac_median_vals))
        res1 = pd.DataFrame(fac_median_vals, index=dat_win_median.index, columns=dat_win_median.columns)
        res2 = datdf[facs_unchanged]
        res = pd.concat([res1, res2], axis=1)
    else:
        # 通过两个repeat，得到与fac_vals 中元素一一对应的极值
        dm = np.repeat(dm.reshape(1, -1), fac_vals.shape[0], axis=0)
        dm1 = np.repeat(dm1.reshape(1, -1), fac_vals.shape[0], axis=0)
        # 替换
        fac_vals = np.where(fac_vals > dm + n*dm1, dm+n*dm1, 
              np.where(fac_vals < dm - n*dm1, dm - n*dm1, fac_vals))
        res = pd.DataFrame(fac_vals, index=dat_win.index, columns=dat_win.columns)

    datdf[facs_to_win] = res
    return datdf  


def neutralize(data, ind_neu=True, size_neu=True, ind='sw', plate=None):
    """
    中性化：因子暴露度对行业哑变量（ind_dummy_matrix）和对数流通市值（lncap_barra）
            做线性回归, 取残差作为新的因子暴露度
    """
    global info_cols
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['Industry_sw']), :]

    tmp_info_cols = [inf for inf in info_cols if inf in datdf.columns]

    # 剔除 info_cols 这些列后剩下的列名
    cols_to_neu = datdf.columns.difference(set(tmp_info_cols))
    y = datdf[cols_to_neu]
    # 剔除含有nan的
    y = y.dropna(how='any', axis=1)
    cols_neu = y.columns

    if size_neu:
        # 对数市值
        lncap = np.log(datdf[['Mkt_cap_float']])

    # 若针对特定行业，则无需生成行业哑变量
    use_dummies = 1

    if not ind_neu:
        use_dummies = 0

    # 市值中性行业不中性
    if use_dummies == 0 and size_neu:
        X = lncap
    # 行业中性市值不中性
    elif use_dummies == 1 and not size_neu:
        X = pd.get_dummies(datdf[f'Industry_{ind}'])
    else:
        # 使用 pd.get_dummies 生成行业哑变量
        ind_dummy_matrix = pd.get_dummies(datdf[f'Industry_{ind}'])
        # 合并对数市值和行业哑变量
        X = pd.concat([lncap, ind_dummy_matrix], axis=1)

    model = LinearRegression(fit_intercept=False)
    # 一次对所有的y都做回归
    try:
        res = model.fit(X, y)
    except Exception as e:
        pd.isna(y).sum().sum()
        pd.isna(X).sum().sum()
        for col, se in y.iteritems():
            pd.isna(se).sum()
            (se == -np.inf).sum()
            np.where(se == -np.inf)
            np.where(se == np.inf)
            print(col)
            res = model.fit(X, se)
            print('debug')
    coef = res.coef_
    residue = y - np.dot(X, coef.T)

    # 断言语言， 如果为false则触发错误
    assert len(datdf.index.difference(residue.index)) == 0

    datdf.loc[residue.index, cols_neu] = residue
    return datdf


def standardize(data):
    """
    标准化：Z-score标准化方法，减去均值，除以标准差
    """
    global info_cols
    
    datdf = data.copy()
    tmp_info_cols = [inf for inf in info_cols if inf in datdf.columns]
    facs_to_sta = datdf.columns.difference(set(tmp_info_cols))
    
    dat_sta = np.float64(datdf[facs_to_sta].values)
    dat_sta = (dat_sta - np.mean(dat_sta, axis=0)) / np.std(dat_sta, axis=0)

    datdf.loc[:, facs_to_sta] = dat_sta
    return datdf


def process_input_names(factor_names):
    if factor_names == 'a':
        factor_names = None
    else:
        factor_names = [f.replace("'", "").replace('"', "") for f in factor_names.split(',')]
    return factor_names


# 向现有的月度因子数据中添加一列因子
def add_columns(added_date_path, columns_list, target_date_path):
    '''
    :param added_date_path:     添加数据的存储位置
    :param columns_list:        准备添加的列名
    :param target_date_path:    需要被添加的数据存储位置
    :return:
    '''
    toadded_list = os.listdir(added_date_path)
    save_list = os.listdir(target_date_path)

    if pd.to_datetime(toadded_list[0].split('.')[0]) > pd.to_datetime(save_list[0].split('.')[0]) or \
            pd.to_datetime(toadded_list[-1].split('.')[0]) < pd.to_datetime(save_list[-1].split('.')[0]):
        print('被添加数据长度不够')
        raise Exception

    for panel_f in os.listdir(target_date_path):
        toadded_dat = pd.read_csv(os.path.join(added_date_path, panel_f),
                                  encoding='gbk', engine='python',
                                  index_col=['code'])

        panel_dat = pd.read_csv(os.path.join(target_date_path, panel_f),
                                encoding='gbk', engine='python',
                                index_col=['code'])

        real_add_list = [col for col in columns_list if col not in panel_dat.columns]
        if len(real_add_list) == 0:
            continue

        # join_axes关键字为沿用那个的index,忽略另一个df的其余数据
        panel_dat = pd.concat([panel_dat, toadded_dat[real_add_list]], axis=1, join_axes=[panel_dat.index])
        panel_dat.to_csv(os.path.join(target_date_path, panel_f),
                         encoding='gbk')

    print('数据添加完毕')


# 根据给定的日度日期序列和月末日期，找到该序列中该月末日期的月初日期
def getmonthfirstdate(dt, md):
    tmp1 = dt[dt.year == md.year]
    tmp2 = tmp1[tmp1.month == md.month]
    return tmp2[0]


# 得到给定日度时间序列的月末时间list
def get_monthends_series(dt):
    if isinstance(dt, pd.DataFrame):
        dt = list(dt)

    p = 0
    med = []
    for i in range(len(dt)-1):
        mon_t = dt[i].month
        mon_n = dt[i+1].month
        if mon_t != mon_n:
            med.append(dt[i])
            p = p + 1

    return pd.Series(med)


def simple_func(pd_s, mv, type='median'):
    # 市值加权
    if type == 'mv_weighted':
        mv_weights = mv/np.sum(mv)
        v = np.dot(np.mat(pd_s), np.mat(mv_weights).T)
        return np.array(v).flatten()
    # 中位数
    elif type == 'median':
        return np.nanmedian(pd_s)
    elif type == 'mean':
        return np.nanmean(pd_s)
    else:
        raise Exception


def apply_func(df, mv, type='median'):
    # 市值加权
    if type == 'mv_weighted':
        mv_weights = mv/np.sum(mv)
        v = np.dot(np.mat(df), np.mat(mv_weights).T)
        return np.array(v).flatten()
    # 中位数
    elif type == 'median':
        return df.median()
    else:
        raise Exception


def apply_func2(x):
    if isinstance(x, str):
        try:
            x = float(x)
        except Exception as e:
            x = 0
    else:
        x
    return x


def concat_factor_2(data_path, save_path, classified_df, factor_name, wei_type, save_name):
    # 创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cols = set(list(classified_df[classified_df.columns[0]]))

    total_df = pd.DataFrame()

    for panel_f in os.listdir(data_path):
        print(panel_f)
        panel_dat = pd.read_csv(os.path.join(data_path, panel_f),
                                encoding='gbk', engine='python',
                                index_col=['code'])

        tmp_df = pd.concat([panel_dat[[factor_name, 'MKT_CAP_FLOAT']], classified_df], axis=1, join='inner')

        d = datetime.strptime(panel_f.split('.')[0], "%Y-%m-%d")
        section_df = pd.DataFrame(index=[d], columns=cols)

        grouped = tmp_df.groupby(classified_df.columns[0])
        for pla, group in grouped:
            group.dropna(how='any', inplace=True)
            section_df.loc[d, pla] = simple_func(group[factor_name], mv=group['MKT_CAP_FLOAT'], type='mv_weighted')[0]

        total_df = pd.concat([total_df, section_df], axis=0)

    if '.' not in save_name:
        save_name = save_name + '.csv'

    total_df.index.name = 'date'

    total_df.to_csv(os.path.join(save_path, save_name), encoding='gbk')
    # 做一个累计净值走势图
    # prod_total_df = (total_df + 1).cumprod()
    # prod_total_df.to_csv(os.path.join(save_path, '累计_'+save_name), encoding='gbk')


# 行业因子合成
def concat_indus_factor(data_path, indus_save_path, compose_way):

    # 创建文件夹
    if not os.path.exists(os.path.join(indus_save_path)):
        os.makedirs(os.path.join(indus_save_path))

    fls = os.listdir(data_path)
    processed_list = os.listdir(indus_save_path)
    to_process_f = [f for f in fls if f not in processed_list]

    if len(to_process_f) == 0:
        print('无需要处理的数据')
        return None

    # 读取行业信息
    # path = r'D:\pythoncode\IndexEnhancement\barra_cne6\basic\industry_citic.csv'
    # industry_df = pd.read_csv(path, encoding='gbk')
    # industry_df.set_index('CODE', inplace=True)

    for panel_f in to_process_f:
        print(panel_f)
        # panel_f = os.listdir(date_path)[0]
        panel_dat = pd.read_csv(os.path.join(data_path, panel_f),
                                encoding='gbk', engine='python',
                                index_col=['Code'])

        # 需要先对股票因子做两个常规处理
        data_to_process, data_unchanged = get_factor_data(panel_dat)
        # data_to_process.empty
        data_to_process = winsorize(data_to_process)

        factors_to_concat = list((set(panel_dat.columns) - (set(info_cols) - set(['Pct_chg_nm']))))
        grouped = data_to_process.groupby('Second_industry')

        ind_factor = pd.DataFrame()
        for name, group in grouped:
            # 选择对应的目标因子
            factor_dat = group[factors_to_concat]
            mv = group['Mkt_cap_float']
            factor_dat = factor_dat.applymap(apply_func2)
            factor_concated = {}
            for factor_name, factors in factor_dat.iteritems():
                if factor_name == 'Lncap_barra':
                    tmp_f = np.log(np.sum(group['Mkt_cap_float']))
                    factor_concated.update({factor_name: tmp_f})
                    continue

                # 不同类型因子有不同的合成方式
                factor_concat_way = 'mv_weighted'
                for concat_way, factorlist in compose_way.items():
                    factorlist_tmp = [fa.lower() for fa in factorlist]
                    if factor_name.lower() in factorlist_tmp:
                        factor_concat_way = concat_way
                tmp_f = simple_func(factors, mv=group['Mkt_cap_float'], type=factor_concat_way)

                factor_concated.update({factor_name: tmp_f})

            factor_concated = pd.DataFrame(factor_concated)
            factor_concated.index = [name]
            factor_concated.loc[name, 'Mkt_cap_float'] = np.sum(mv)                   # 市值采用行业市值和
            if 'Industry_zx' in group.columns:
                factor_concated.loc[name, 'Industry_zx'] = group.loc[group.index[0], 'Industry_zx']
            if 'Industry_sw' in group.columns:
                factor_concated.loc[name, 'Industry_sw'] = group.loc[group.index[0], 'Industry_sw']
            ind_factor = pd.concat([ind_factor, factor_concated], axis=0)

        ind_factor.index.name = 'Name'
        ind_factor.to_csv(os.path.join(indus_save_path, panel_f), encoding='gbk')


# 把一个 截面数据添加到已经有的月度模式存储的文件中
def add_to_panels(dat, panel_path, f_name, freq_in_dat='M'):
    """说明： 把dat依次插入到panel_path的DF中，插入的列名为f_name, 根据dat的类型是DF还是Series可以判断
    是每次插入的数据不同还是每次插入相同的数据。"""

    print(f'开始添加{f_name}数据到目标文件夹')
    panel = os.listdir(panel_path)
    for month_date in panel:
        hased_dat = pd.read_csv(os.path.join(panel_path, month_date), engine='python')
        hased_dat = hased_dat.set_index('Code')

        # 输入数据为 DataFrame, 那么按列插入
        if isinstance(dat, pd.DataFrame):
            mon_str = month_date.split('.')[0]
            if mon_str in dat.columns:
                # 当dat中的columns也是str格式，且日期与panel一样时，直接添加
                hased_dat[f_name] = dat[mon_str]
            else:
                # 否则，当年、月相同，日不同时，需要变成datetime格式而且还有查找
                target = datetime.strptime(mon_str, "%Y-%m-%d")
                # 当dat的columns是datetime格式时
                if isinstance(dat.columns[0], datetime):
                    if freq_in_dat == 'M':
                        finded = None
                        for col in dat.columns:
                            if col.year == target.year and col.month == target.month:
                                finded = col
                                break
                        if finded:
                            hased_dat[f_name] = dat[finded]
                        else:
                            print('{}该期未找到对应数据'.format(mon_str))
                    if freq_in_dat == 'D':
                        if target in dat.columns:
                            hased_dat[f_name] = dat[target]
                        else:
                            print('{}该期未找到对应数据'.format(mon_str))
                else:
                    print('现有格式的还未完善')
                    raise Exception
        # 输入数据为 DataFrame, 那么按列插入
        elif isinstance(dat, pd.Series):
            hased_dat[f_name] = dat[hased_dat.index]

        try:
            hased_dat = hased_dat.reset_index('Code')
        except Exception as e:
            print('debug')

        if 'No' in hased_dat.columns:
            del hased_dat['No']
        hased_dat.index.name = 'No'
        hased_dat.to_csv(os.path.join(panel_path, month_date), encoding='gbk')

    print('完毕！')


def adjust_months(d_df):

    if isinstance(d_df.columns[0], str):
        new_cols = [datetime.strptime(col, "%Y-%m-%d") for col in d_df.columns]
        d_df.columns = new_cols

    # 删除12月份的数据
    tdc = [col for col in d_df.columns if col.month == 12]
    d_df = d_df.drop(tdc, axis=1)

    # 把公告月份调整为实际月份
    new_cols = []
    for col in d_df.columns:
        if col.month == 3:
            new_cols.append(datetime(col.year, 4, 30))
        if col.month == 6:
            new_cols.append(datetime(col.year, 8, 31))
        if col.month == 9:
            new_cols.append(datetime(col.year, 10, 31))

    d_df.columns = new_cols

    return d_df


# 从一个月度panel里面删除某个因子
def del_factor_from_panel(panel_path, factor_name):

    print(f'开始从目标文件夹删除{factor_name}因子。')
    panel = os.listdir(panel_path)
    for month_date in panel:
        dat_df = pd.read_csv(os.path.join(panel_path, month_date), engine='python')
        dat_df = dat_df.set_index('Code')

        if factor_name in dat_df.columns:
            del dat_df[factor_name]
            dat_df.reset_index(inplace=True)
            dat_df.set_index('No', inplace=True)
            dat_df.index.name = 'No'
            dat_df.to_csv(os.path.join(panel_path, month_date), encoding='gbk')

    print(f'完毕。')
    return


# 根据 months_end 的列拓展 d_df。
# 我从数据库中下载的基本面数据是调整过日期的，但是调整到的是月度的最后一个自然日，要改成月度最后一个交易日，
# 同时还有复制到其他月份中。
def append_df(d_df, target_feq='M', fill_type='preceding'):
    '''
    fill_type ：若整列和为0的填充方式，preceding表示使用前值填充,  empty表示不填充
    '''

    tds = trade_days()

    # 得到月末日期列表
    months_end = []
    for i in range(1, len(tds)):
        if tds[i].month != tds[i - 1].month:
            months_end.append(tds[i - 1])
        elif i == len(tds) - 1:
            months_end.append(tds[i])
    try:
        months_end = [me for me in months_end if me.year >= d_df.columns[0].year]
    except Exception as e:
        print('debug')

    # 赵到对应的月末日期列表，可能年月同日不同的情况
    new_col = []
    for col in d_df.columns:
        for me in months_end:
            if col.year == me.year and col.month == me.month:
                new_col.append(me)
    # 改变月末日期
    d_df.columns = new_col

    if target_feq.upper() == 'M':
        # 设一个日期全的，单值为空的df
        res = pd.DataFrame(index=d_df.index, columns=months_end)
        # 给定日期赋值
        res[d_df.columns] = d_df

    elif target_feq.upper() == 'D':
        new_columns = [d for d in tds if d >= d_df.columns[0]]
        res = pd.DataFrame(index=d_df.index, columns=new_columns)
        # 给定日期赋值
        res[d_df.columns] = d_df

    elif target_feq.upper() == 'W':
        week_ends = trade_days('w')
        # 因为 月末交易日数据（A） 与 一周交易日数据最后一天(B) 不是一一对应也不是B包含A的关系，所以要做一个A与相对应的B的映射
        res = pd.DataFrame(index=d_df.index, columns=week_ends)
        selected_cols = []
        for col, se in d_df.iteritems():
            delta = [we - col for we in week_ends if (we - col).days >= 0]
            selected_cols.append(col + np.min(delta))
        # 给定日期赋值
        res[selected_cols] = d_df

    # 首期赋值为None
    if res.iloc[:, 0].sum() == 0:
        res.iloc[:, 0] = np.nan

    # 若当列为空，则当列数值与前列相同
    if fill_type == 'preceding':
        res_ar = np.array(res)
        [h, l] = res_ar.shape
        for i in range(1, l):
            for j in range(0, h):
                if np.isnan(res_ar[j, i]) and not np.isnan(res_ar[j, i-1]):
                    res_ar[j, i] = res_ar[j, i-1]

        res_df = pd.DataFrame(data=res_ar, index=res.index, columns=res.columns)

    # 删除nan
    res_df.dropna(axis=1, how='all', inplace=True)
    res_df.dropna(axis=0, how='all', inplace=True)
    res_df.sum()

    return res_df

if __name__ == "__main__":

    panel_path = r"D:\pythoncode\IndexEnhancement\因子预处理模块\因子"
    factor_name_list = ['Totaloperatingrevenueps_qoq_qoq']
    for f in factor_name_list:
        del_factor_from_panel(panel_path, f)

    # panel_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    # add_fs_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\增加的因子\截面数据'
    #
    # f_list = os.listdir(add_fs_path)
    # for fn in f_list:
    #     f_name = fn.split('.')[0]
    #     print(f_name)
    #     dat = pd.read_csv(os.path.join(add_fs_path, fn), engine='python')
    #     dat = dat.set_index(dat.columns[0])
    #     add_to_panels(dat, panel_path, f_name)



