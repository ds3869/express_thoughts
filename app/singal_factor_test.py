# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:00:11 2019

@author: HP
"""
import os
import warnings
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from utility.index_enhance import get_factor, get_factor_corr, plot_corr_heatmap
from utility.constant import plate_to_indus, factor_path, sf_test_save_path
from utility.single_factor_test import get_firt_industry_list


warnings.filterwarnings('ignore')  # 将运行中的警告信息设置为“忽略”，从而不在控制台显示

os.chdir(sf_test_save_path)
from utility.single_factor_test import (test_yearly, get_factor_names, panel_to_matrix,
                                        SingleFactorLayerDivisionBacktest, plot_layerdivision, bar_plot_yearly,
                                        plot_group_diff_plot)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)  # 图片尺寸设定（宽 * 高 cm^2)

err_industry = defaultdict(list)  # 记录下有问题的行业与因子


def single_factor_test(factors, path_dict, ind_benchmark=None):
    '''
    plate为一个tuple, 0为板块名称，1为其包含得一级行业，如果是单行业，那么板块名称和行业相同
    '''

    print("\n开始进行T检验和IC检验...")
    ny = datetime.today().year
    test_yearly(path_dict, factors, end_year=ny, industry_benchmark=ind_benchmark)   # T检验&IC检验n
    print(f"检验完毕！结果见目录：{path_dict['sf_test_save_path']}")
    print('*'*80)


def form_panle(factors, path_dict, sp=None):

    factor_p = path_dict['factor_path']
    factor_matrix_p = path_dict['factor_matrix_path']

    if not os.path.exists(factor_matrix_p):
        os.makedirs(factor_matrix_p)

    # 创建因子矩阵文件，为分层回测做准备
    panel_to_matrix(factors, factor_path=factor_p, save_path=factor_matrix_p, special_plate=sp)


def layer_division_bt(factors, path_d, layer_num):

    sf_test_save_path = path_d['sf_test_save_path']
    factor_path = path_d['factor_path']
    factor_matrix_path = path_d['factor_matrix_path']

    start_date = '2009-02-27'
    end_date = '2019-07-31'
    if_concise = True          # 是否进行月频简化回测

    save_path_tmp = os.path.join(sf_test_save_path, '分层回测')
    # 创建分层回测结果图的存放目录
    if not os.path.exists(save_path_tmp):
        os.mkdir(save_path_tmp)

    # print('因子数据创建完毕')
    pct_chg_nm = get_factor(['PCT_CHG_NM'], basic_path=factor_matrix_path)['PCT_CHG_NM']

    # 对选中的因子或者全部因子遍历
    # print("开始进行因子分层回测...")
    for fname in factors:
        # print(fname)
        openname = fname.replace('/', '_div_')
        facdat = pd.read_csv(os.path.join(factor_matrix_path, openname+'.csv'),
                             encoding='gbk', engine='python', index_col=[0])
        facdat.columns = pd.to_datetime(facdat.columns)

        s = SingleFactorLayerDivisionBacktest(factor_name=fname,
                                              factor_data=facdat,
                                              num_layers=layer_num,
                                              if_concise=if_concise,
                                              start_date=start_date,
                                              end_date=end_date,
                                              pct_chg_nm=pct_chg_nm)

        records = s.run_layer_division_bt(equal_weight=True)

        if not records.empty:
            plot_layerdivision(path_d, records, fname, if_concise)         # 绘制分层图
            bar_plot_yearly(path_d, records, fname, if_concise)            # 绘制分年分层收益柱形图
            plot_group_diff_plot(path_d, records, fname, if_concise)       # 绘制组1-组5净值图

    print(f"分层回测结束！结果见目录：{sf_test_save_path}")

    print('*'*80)


def main(path_dict, task=None, layer_num=5, factors=None):

    if not factors:
        factors = get_factor_names(path_dict['factor_path'])

    if not os.path.exists(path_dict['sf_test_save_path']):
        os.makedirs(path_dict['sf_test_save_path'])

    # 对所有股票做测试
    if not task:        # 若不指定，则全部做
        single_factor_test(factors, path_dict)
        form_panle(factors, path_dict)
        layer_division_bt(factors, path_dict, layer_num)
    else:
        if 'panel' in task:
            form_panle(factors, path_dict)

        if 'st' in task:
            single_factor_test(factors, path_dict)

        if 'bt' in task:
            layer_division_bt(factors, path_dict, layer_num)


def update_industry_select():
    path_dict = {'factor_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\因子(已预处理)',
                 'sf_test_save_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验',
                 'factor_matrix_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验\因子矩阵',
                 'total_result_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\结果汇总比较',
                 }
    factors = get_factor_names(path_dict['factor_path'])
    form_panle(factors, path_dict)


if __name__ == '__main__':

    special_plate = None
    test_type = 'stock'

    # 分二级行业做（已废弃）
    if test_type == 'second_industry':
        layer_num = 2
        path_dict = {'factor_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\因子(已预处理)',
                     'sf_test_save_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验',
                     'factor_matrix_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验\因子矩阵',
                     'total_result_path': r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\结果汇总比较',
                     }
        special_plate = None
        task = None
        main(path_dict, task, layer_num)
    # 股票
    elif test_type == 'stock':
        layer_num = 5
        path_dict = {'factor_path': r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）',
                     'sf_test_save_path': r'D:\pythoncode\IndexEnhancement\单因子检验',
                     'factor_matrix_path': r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵',
                     'total_result_path': r'D:\pythoncode\IndexEnhancement\单因子检验\结果汇总比较',
                     }
        special_plate = None
        task = ['panel']
        main(path_dict, task, layer_num)
    # 单行业测试
    elif test_type == 'each_industry':
        layer_num = 3
        indus_list = get_firt_industry_list()
        # for indus in indus_list:
        if True:
            # indus = '建筑材料'
            # special_plate = '建筑材料'
            special_plate = '建筑材料'
            print(special_plate)
            layer_num = 3
            path_dict = {'factor_path': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     '分行业研究', special_plate, '因子（已预处理）').replace('\\', '/'),
                         'sf_test_save_path': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                           '分行业研究', special_plate, '单因子检验').replace('\\', '/'),
                         'factor_matrix_path': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                            '分行业研究', special_plate, '单因子检验',
                                                            '因子矩阵').replace('\\', '/'),
                         'total_result_path': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                           '分行业研究', special_plate, '单因子检验',
                                                           '结果汇总比较').replace('\\', '/'),
                         }
            main(path_dict, None, layer_num)
            print('{}行业的因子截面数据已全部处理！'.format(special_plate))


    # # 因子相关性检验
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # plt.rcParams['figure.figsize'] = (16.0, 9.0)  # 图片尺寸设定（宽 * 高 cm^2)
    # plt.rcParams['font.size'] = 15  # 字体大小
    #
    # sn = '因子热力图'
    # sp = r'D:\pythoncode\IndexEnhancement\单因子检验'
    # factors_to_corr = ['SUE', 'REVSU', 'Sales_G_q', 'Profit_G_q', 'return_1m', 'return_3m']
    # ori_factor_path = r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'
    # corr_ori = get_factor_corr(factors_to_corr, basic_path=ori_factor_path)
    # plot_corr_heatmap(corr_ori, save_path=sp, save_name=sn)
