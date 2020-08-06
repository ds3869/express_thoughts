'''
提取第二行业、第三行业因子，然后存储成月度数据模式。再与原来的月度因子合并。
以后如果再添加别的因子，也按照该流程。
'''
from collections import defaultdict
import pandas as pd
import numpy as np
import re
from barra_cne6.barra_template import Data
from utility.factor_data_preprocess import get_monthends_series, info_cols, concat_indus_factor, add_to_panels,\
                                           concat_factor_2
import os
import shutil
from utility.constant import plate_to_indus


def add_industry_infor():
    # 存储临时数据地址
    tmp_save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\tmp'
    if not os.path.exists(tmp_save_path):
        os.makedirs(tmp_save_path)

    # 获取数据
    data = Data()
    second_ind = data.industry_sw_2
    # pd.DataFrame(set(second_ind[second_ind.columns[0]].values)).to_csv(r'D:\pythoncode\IndexEnhancement\二级行业名称.csv',
    #                                                                    encoding='gbk')

    # 输入的类型为Series
    inps = second_ind[second_ind.columns[0]]
    # 与原月度数据合并
    target_date_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    add_to_panels(inps, target_date_path, 'second_industry')


# 生成行业因子数据
def generate_indus_factor():

    # 生成行业因子
    data_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    indus_save_path = r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\因子'

    median_factors_0 = ['EP', 'EPcut', 'BP', 'SP', 'NCFP', 'OCFP', 'DP', 'G/PE', 'Sales_G_q', 'Profit_G_q',
                      'OCF_G_q', 'ROE_G_q', 'ROE_q', 'ROE_ttm', 'ROA_q', 'ROA_ttm', 'grossprofitmargin_q',
                      'grossprofitmargin_ttm', 'profitmargin_q', 'profitmargin_ttm', 'assetturnover_q',
                      'assetturnover_ttm', 'operationcashflowratio_q', 'operationcashflowratio_ttm',
                      'financial_leverage', 'debtequityratio', 'cashratio', 'currentratio', 'ln_capital',
                      'HAlpha', 'netprofitratiottm_qoqgt', 'opercashpsgrowrate', 'operprofitgrowrate',
                      'roettm_qoqgt', 'exp_to_rev', 'exp_to_rev_yoy', 'SUE', 'REVSU', 'roettm_yoy',
                      'netprofitratiottm_yoy', 'operatingrevenuepsttm_yoy', 'totaloperatingrevenueps_qoq_qoq',
                      'PEG']

    median_factors_1 = []
    for col in median_factors_0:
        new_c = col[0].upper() + col[1:].lower()
        median_factors_1.append(new_c)

    median_factors = median_factors_0 + median_factors_1

    compose_way = {'median': median_factors}
    concat_indus_factor(data_path, indus_save_path, compose_way)


def generate_palte_pct(plate_to_indus_dict):
    data_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    indus_save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\market_quote'
    factor_name = 'PCT_CHG_NM'
    wei_type = 'mv_weighted'
    save_name = 'PLATE_PCT_CHG_NM'

    # plate_to_indus_2 = {'周期': ['石油石化', '采掘', '有色金属', '化工', '钢铁', '建筑材料', '建筑装饰'],
    #                     '中游': ['轻工制造', '国防军工', '公用事业', '交通运输', '综合'],
    #                     '消费': ['食品饮料', '医药生物', '农林牧渔', '汽车', '家用电器', '休闲服务', '纺织服装', '商业贸易'],
    #                     '成长': ['电子', '计算机', '通信', '传媒', '机械设备', '电气设备'],
    #                     # '金融地产': ['银行', '非银金融', '房地产'],
    #                     }

    indus_to_plate = {}
    for key, value in plate_to_indus_dict.items():
        for v in value:
            indus_to_plate.update({v: key})

    industry_citic = pd.read_csv(r'D:\pythoncode\IndexEnhancement\barra_cne6\basic\industry_citic.csv', encoding='gbk')
    industry_citic.set_index('CODE', inplace=True)
    industry_citic['plate'] = np.nan
    df_tmp = pd.DataFrame()
    for i, r in industry_citic.iterrows():
        if r['中信一级行业'] in indus_to_plate.keys():
            r['plate'] = indus_to_plate[r['中信一级行业']]
            df_tmp = pd.concat([df_tmp, pd.DataFrame({i: r}).T], axis=0)

    classified_df = pd.DataFrame({'plate': df_tmp['plate']})

    concat_factor_2(data_path=data_path, save_path=indus_save_path, classified_df=classified_df, factor_name=factor_name,
                    wei_type=wei_type, save_name=save_name)


if __name__ == '__main__':
    # 在原始因子上添加行业信息
    # add_industry_infor()
    generate_indus_factor()

    # generate_palte_pct(plate_to_indus)










