#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from datetime import datetime
import shutil
from sklearn.covariance import LedoitWolf
from utility.factor_data_preprocess import adjust_months, add_to_panels, align, append_df
from utility.tool0 import Data, add_stock_pool_txt
from utility.relate_to_tushare import stocks_basis, generate_months_ends
from utility.stock_pool import financial_condition_pool, factor_condition_pool, concat_stock_pool, save_each_sec_name,\
    del_industry, keep_industry, get_scores, twice_sort, keep_market, from_stock_wei_2_industry_wei
from utility.download_from_wind import section_stock_infor
from utility.analysis import BackTest, bool_2_ones
from utility.stock_pool import float_2_bool_df, cond_append_to_month, month_return_compare_to_market_index
from utility.select_industry import my_factor_concat, history_factor_return,  forecast_factor_return, \
    copy_matrix, forecast_factor_return
from utility.index_enhance import linear_programming, concat_factors_panel, get_factor, get_est_stock_return
from utility.optimization import optimization_fun



class FormStockPool:
    """
    Setting for runnning optimization.
    """

    def __init__(self, abs_financial_dict, factor_wei, opt_para_dict, factor_dict=None, path_dict=None,
                 alpha_factor=None, risk_factor=None, select_mode='total_num',
                 first_max=50, twice_sort_dict={}, max_n=20, freq='M', industry_dict=None, special_market=None,
                 benchmark='ZZ500', method='regress'):
        """"""

        self._data = Data()

        # 回测开始日期
        self.start_date = datetime(2009, 3, 1)
        # 回测结束日期
        self.end_date = None
        # 选股方法： 打分法还是回归法
        self.rov_method = method

        # 财务绝对要求条件
        self.abs_financial_dict = abs_financial_dict
        # 所有因子
        if not factor_dict:
            self.factor_dict = {
                'mom': ['Return_12m', 'Return_1m', 'Return_3m', 'Return_6m'],
                'quality': ['Roa_q', 'Roe_q'],
                'value': ['Ep'],
                'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q'],
                'size': ['Lncap_barra'],
                'liquidity': ['Stom_barra', 'Stoq_barra', 'Stoa_barra'],
                'volatility': ['Std_1m', 'Std_3m', 'Std_6m', 'Std_12m'],
            }
        else:
            self.factor_dict = factor_dict

        # 因子合成地址
        if not path_dict:
            self.path_dict = {'save_path': r'D:\pythoncode\IndexEnhancement\多因子选股',
                              'factor_panel_path': r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）',
                              'ic_path': r'D:\pythoncode\IndexEnhancement\单因子检验\ic.csv',
                              'old_matrix_path': r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'}
        else:
            self.path_dict = path_dict

        # alpha因子, self.factor_dict中的key且不再alpha的名称列表里的，是风险因子
        self.alpha_factor = alpha_factor
        if not alpha_factor:
            self.alpha_factor = ['quality', 'growth']
        else:
            self.alpha_factor = alpha_factor

        self.risk_factor = risk_factor
        self.factor_wei = factor_wei
        self.opt_para_dict = opt_para_dict
        self.factors_panel = None

        # 若是两步选择法，第一次选择股票测数量
        self.first_max = first_max
        # 总股票数量
        self.max_n = max_n
        # 单个行业的股票数量
        self.one_industry_num = 5
        # 因子加权方式为等权
        self.wei_type = 'equal'
        # 测试频率
        self.freq = freq
        # 行业设定
        self.industry_dict = industry_dict
        # 特定板块设定
        self.special_market = special_market
        # 历史的因子表现
        self.factor_ret_history = None
        # 股票的特质收益率
        self.resid = None
        # 计算压缩矩阵时的窗口期
        self.shrinkaged_window = 12
        # 不同截面的组合协方差矩阵
        self.E_dict = None

        self.benchmark = benchmark
        self.index_wei = None
        self.constituent_industry = None
        self.set_benchmark()

        self.pool = None
        stock_basic = self._data.stock_basic_inform
        self.industry_map = pd.DataFrame({'first_industry': stock_basic['申万一级行业']})

        # 设置为'total_num'，表示从所有股票中选择排名靠前的， 'by_industry'表示分行业选择, twice_sort表示两步排序法
        self.select_type = select_mode
        self.twice_sort_dict = twice_sort_dict

    def set_benchmark(self):
        if self.benchmark == 'allA':  # 'HS300' 'ZZ500' 'SZ50'
            self.constituent_industry = None
        elif self.benchmark == 'HS300':
            tmp = self._data.hs300_wt
            tmp = tmp / 100
            self.index_wei = tmp
            self.constituent_industry = from_stock_wei_2_industry_wei(tmp)
        elif self.benchmark == 'ZZ500':
            tmp = self._data.zz500_wt
            tmp = tmp / 100
            self.index_wei = tmp
            self.constituent_industry = from_stock_wei_2_industry_wei(tmp)

    def rov_of_all_stocks(self, method):
        # 通过打分法选股, stock_pool 为各个股票的打分
        if method == 'score':
            stock_pool = self.select_stocks_by_scores()
        # 通过回归法选股，stock_pool 为各个股票的预测收益率
        elif method == 'regress':
            stock_pool = self.select_stocks_by_regress()

        return stock_pool

    def select_stocks_by_regress(self):

        # 预测的因子收益
        est_facor_rets = forecast_factor_return(self.alpha_factor, self.factor_ret_history, window=12)
        # 全部股票收益预测
        est_stock_rets = get_est_stock_return(self.alpha_factor, self.factors_panel, est_facor_rets, 12, 6)
        print('计算预期收益率完成...')

        est_stock_rets.name = 'stock_return'

        if isinstance(self.pool, pd.DataFrame):
            # 使用财务数据过滤一下股票池
            self.pool
            ret = est_stock_rets
        else:
            ret = est_stock_rets

        return est_stock_rets

    # 风险模型（组合优化）
    def compose_optimization(self, stock_pool, is_enhance=True, lamda=10, turnover=None, te=None, industry_max_expose=0,
                             size_neutral=True, in_benchmark=False, limit_factor_panel=None, max_num=None):
        '''
        :param stock_pool:    每期的alpha排序或预测的股票收益
        :param is_enhance:    是否是指数增强
        :param lamda:         风险厌恶系数
        :param turnover:      是否有换手率约束
        :param te:            是否有跟踪误差约束
        :param industry_max_expose  行业风险敞口，如果该值为0，则表示行业中性
        :param size_neutral   是否做市值中性
        :param in_benchmark:  是否必须成份股内选股
        :param max_num:       最大股票数量要求
        :return: stock_wei:   每期的股票权重
        '''

        his_tamp = list(set(stock_pool.columns) & set(self.E_dict.keys()))
        his_tamp.sort()
        wei_df = pd.DataFrame()
        save_path = r'D:\pythoncode\IndexEnhancement\股票池_临时'
        pre_w = None
        # todo 添加一个读取已经优化好的权重数据的功能。
        for d in his_tamp:

            # if d < datetime(2014, 4, 30):
            #     continue

            # if d == datetime(2011, 11, 30):
            #     print('debug')
            # d = pd.Timestamp(datetime(2010, 9, 30))

            loc = his_tamp.index(d)
            # 如果
            if loc != 0 and not isinstance(pre_w, pd.Series):
                his_wei = pd.read_csv(os.path.join(save_path, '股票权重.csv'), engine='python')
                his_wei.set_index(his_wei.columns[0], inplace=True)
                his_wei.columns = pd.to_datetime(his_wei.columns)
                pre_w = his_wei[his_tamp[loc-1]]

            print(d)
            r_tmp = stock_pool[d]
            e_tmp = self.E_dict[d]
            bench_wei = self.index_wei[d].dropna()
            f_n = self.risk_factor.keys()
            limit_factor_df = self.factors_panel[d][f_n]
            in_benchmark = in_benchmark

            wei = optimization_fun(r_tmp, e_tmp, bench_wei, pre_w=pre_w, lamda=lamda, turnover=turnover, te=te,
                                   industry_max_expose=industry_max_expose, risk_factor_dict=self.risk_factor,
                                   limit_factor_df=limit_factor_df, in_benchmark=in_benchmark,
                                   max_num=max_num)
            pre_w = wei
            wei_df = pd.concat([wei_df, pd.DataFrame({d: wei})], axis=1)
            wei_df.to_csv(os.path.join(save_path, '股票权重.csv'), encoding='gbk')

        print('全部优化完毕')
        return wei_df

    def easy_compose_optimization(self, stock_pool, bench, ):
        # 得到风险因子的因子矩阵，以dict形式存储，key为因子名称。
        risk_factors = ['size']      # self.factor_dict['risk_factor']
        codes = list(stock_pool.index)
        risk_fac_data = {fac: get_factor([fac], codes)[fac] for fac in risk_factors}

        # 把因子矩阵形式的存储，变成字典形式的存储，每个key是日期，value是行为codes，列为factors的dataframe
        limit_fac_data = concat_factors_panel(risk_factors, risk_fac_data, codes,
                                              ind=True, mktcap=False)

        data_dict = {'limit_fac_data': limit_fac_data,
                     'index_wt': self.index_wei,
                     'est_stock_rets': stock_pool
                     }
        stock_wt = linear_programming(data_dict)

        return stock_wt

    def compute_factor_return(self):

        factors_dict = self.factor_dict
        path_dict = self.path_dict

        # ----------------------------------
        # 因子检测
        f_list = []
        for values in factors_dict.values():
            for v in values:
                f_list.append(v)
        f_list = [col.replace('_div_', '/') for col in f_list]

        tmp = os.listdir(path_dict['factor_panel_path'])
        df_tmp = pd.read_csv(os.path.join(path_dict['factor_panel_path'], tmp[-1]),
                             engine='python', encoding='gbk', index_col=[0])

        cols = [col for col in df_tmp.columns]

        if not set(f_list).issubset(cols):
            print('factor 不够, 缺失的因子为：')
            print(set(f_list) - set(cols))
        else:
            print('通过因子完备性测试')

        # ----------------------------------
        # 因子合成
        # if os.path.exists(os.path.join(path_dict['save_path'], '新合成因子')):
        #     shutil.rmtree(os.path.join(path_dict['save_path'], '新合成因子'))
        # print('开始进行因子合成处理.....')
        # for factor_con, factors_to_con in factors_dict.items():
        #     # 'equal_weight'   'max_ic_ir'
        #     my_factor_concat(path_dict, factors_to_con, factor_con, concat_type='equal_weight')
        # print('因子合成完毕！')

        params = {
            'factors': [key for key in factors_dict.keys()],
            'window': 6,
            'half_life': None,
        }

        copy_matrix(path_dict['old_matrix_path'], os.path.join(path_dict['save_path'], '新合成因子', '因子矩阵'))

        # 估计预期收益
        path_dict.update({'matrix_path': os.path.join(path_dict['save_path'], '新合成因子', '因子矩阵')})
        matrix_path = path_dict['matrix_path']

        factors, window, half_life = params['factors'], params['window'], \
                                     params['half_life']

        factors.extend(['Pct_chg_nm'])
        # 得到所有因子的因子矩阵，以dict形式存储，行业部分没有风险因子
        factors_dict = {fac: get_factor(factors, basic_path=matrix_path)[fac] for fac in factors}
        # 将alpha因子整理为截面形式
        factors_panel = concat_factors_panel(factors=None, factors_dict=factors_dict, codes=None,
                                             ind=False, mktcap=False, perchg_nm=False)

        # 删除开始factor不全的截面
        to_del = []
        for key, values in factors_panel.items():
            print(key)
            for f in factors:
                if f not in values.columns:
                    print(f)
                    to_del.append(key)
                    break

        for d in to_del:
            factors_panel.pop(d)

        self.factors_panel = factors_panel

        factor_ret_history, sig = history_factor_return(factors, factors_panel, window, half_life)
        # 最后一行的nan先不能删除，因为后面预测时要向后shift一行
        return factor_ret_history, sig

    def get_shrinkaged(self):
        '''
        E = fX*F_shrinkaged*fX.T + e
        :return:
        '''

        shrinkaged_dict = {}
        self.shrinkaged_window = 12

        # 因子名称
        fn = [k for k in self.factor_dict.keys()]

        for l in range(self.shrinkaged_window, len(self.factor_ret_history)):

            factor_ret_tmp = self.factor_ret_history.loc[self.factor_ret_history.index[l-self.shrinkaged_window:l], fn]
            try:
                cov = LedoitWolf().fit(factor_ret_tmp)
                factor_cov_tmp = cov.covariance_  # 通过压缩矩阵算法得到因子收益协方差矩阵
            except Exception as e:
                print('压缩矩阵1')

            # 该期的因子截面数据
            dat_tmp_df = self.factors_panel[self.factor_ret_history.index[l]]
            f_expo = dat_tmp_df.loc[:, fn].dropna(axis=0, how='any')

            # f_expo为N*M，N为股票数量，M为因子个数； factor_cov_tmp为M*M，f_expo.T 为M*N，最后结果为N*N
            factor_cov = np.dot(np.dot(f_expo.values, factor_cov_tmp), f_expo.values.T)
            factor_cov_df = pd.DataFrame(data=factor_cov, index=f_expo.index, columns=f_expo.index)

            resid_tmp = self.resid.loc[self.factor_ret_history.index[l-self.shrinkaged_window:l], :]
            resid_tmp2 = resid_tmp.loc[:, f_expo.index].fillna(0)

            # 两部分相加，就是整个的股票组合的协方差矩阵，在马科维茨模型中就表示风险
            e_ef = factor_cov_df + resid_tmp2.cov()
            shrinkaged_dict.update({self.factor_ret_history.index[l]: e_ef})

        return shrinkaged_dict

    def run_test(self):
        # 负向指标选股，剔除一些不好的股票
        self.get_stock_pool()
        # 历史的因子表现 以及 股票的特质收益率
        self.factor_ret_history, self.resid = self.compute_factor_return()
        self.E_dict = self.get_shrinkaged()

        # 正向选股，打分法或者是回归法，返回股票的打分或者是回归得到的预测收益率，
        stock_pool = self.rov_of_all_stocks(self.rov_method)
        # 根据打分或是预测的收益率，以及基准，得到股票的配置权重
        is_enhance = self.opt_para_dict['is_enhance']
        lamda = self.opt_para_dict['lamda']
        turnover = self.opt_para_dict['turnover']
        te = self.opt_para_dict['te']
        industry_max_expose = self.opt_para_dict['industry_max_expose']
        size_neutral = self.opt_para_dict['size_neutral']
        in_benchmark = self.opt_para_dict['in_benchmark']
        max_num = self.opt_para_dict['max_num']
        stock_pool = self.compose_optimization(stock_pool, lamda=lamda, is_enhance=is_enhance, turnover=turnover, te=te,
                                               industry_max_expose=industry_max_expose, size_neutral=size_neutral,
                                               in_benchmark=in_benchmark, max_num=max_num)

        pp = r'D:\pythoncode\IndexEnhancement\股票池_最终\500增强.csv'
        # stock_pool.to_csv(pp, encoding='gbk')

        stock_pool = pd.read_csv(pp, engine='python', encoding='gbk')
        stock_pool.set_index(stock_pool.columns[0], inplace=True)
        stock_pool.columns = pd.to_datetime(stock_pool.columns)

        ana = self.backtest(stock_pool, bench=self.benchmark, re_invest='cash', hedging=True)   # re_invest = cash or re_weighted

        return ana

    def position_daily(self, stock_pool):
        data = Data()
        main_net_buy_ratio = data.MAIN_NET_BUY_RATIO_SCALERED
        main_net_buy_ratio = main_net_buy_ratio.shift(1, axis=1)
        main_net_buy_ratio = main_net_buy_ratio.drop(main_net_buy_ratio.columns[0], axis=1)

        black_cond = float_2_bool_df(main_net_buy_ratio, min_para=5)
        black_cond = cond_append_to_month(black_cond)

        stock_pool_v = bool_2_ones(stock_pool, use_df=True)
        pool = append_df(stock_pool_v, target_feq='D')
        pool = pool.shift(1, axis=1)
        pool.dropna(how='all', axis=1, inplace=True)

        pool_boll = float_2_bool_df(pool, min_para=0.5)
        # ttt = pool_boll.sum()

        mu_col = [c for c in black_cond.columns if c in pool_boll.columns]

        res = pool_boll.loc[:, mu_col] & black_cond.loc[pool.index, mu_col]
        # tt = res.sum()
        return res

    def get_stock_pool(self):

        # 财务绝对要求条件的股票池构建，如roettm大于5%，eps同比增速大于5，sue位于所有股票的top5%。
        stock_pool = financial_condition_pool(self.abs_financial_dict, self.start_date, self.end_date)

        if self.industry_dict:
            if self.industry_dict['handle_type'] == 'delete':
                stock_pool = del_industry(stock_pool, self.industry_dict['to_handle_indus'])
            elif self.industry_dict['handle_type'] == 'keep':
                stock_pool = keep_industry(stock_pool, self.industry_dict['to_handle_indus'])
        # 特定股票板块
        elif self.special_market:
            dat_df = stocks_basis()
            dat_df = dat_df.set_index('ts_code')
            codes = list(dat_df.index[dat_df['market'] == self.special_market])
            # todo  逻辑上和keep_industry有部分功能是一样的，可以改写一下
            stock_pool = keep_market(stock_pool, codes)     # 只暴露特定的股票

        # 选择给定时间段的列
        cols = [col for col in stock_pool.columns if col >= self.start_date]
        stock_pool = stock_pool[cols]
        stock_pool.sum()
        self.pool = stock_pool

    def select_stocks_by_scores(self):
        new_stock_pool = pd.DataFrame()

        self.pool.sum()
        for col, value in self.pool.iteritems():
            codes = list(value[value == True].index)
            codes_selected = self.section_of_select_stocks_by_scores(codes, col)

            if isinstance(codes_selected, list):
                tmp = pd.DataFrame(np.full(len(codes_selected), True), index=codes_selected, columns=[col])
                new_stock_pool = pd.concat([new_stock_pool, tmp], axis=1)
            elif isinstance(codes_selected, pd.DataFrame):
                new_stock_pool = pd.concat([new_stock_pool, codes_selected], axis=1)

        new_stock_pool.fillna(0, inplace=True)
        return new_stock_pool

    def section_of_select_stocks_by_scores(self, codes, dt):
        factor_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
        f_list = os.listdir(factor_path)
        fn = dt.strftime("%Y-%m-%d") + '.csv'

        if fn not in f_list:
            # print('未找到'+ fn + '该期数据')
            return None
        data = pd.read_csv(os.path.join(factor_path, fn), engine='python', encoding='gbk')
        data = data.set_index('Code')
        new_index = [c for c in codes if c in data.index]
        data = data.loc[new_index, :]

        constituent_se = self.constituent_industry[dt]

        # 根据因子字典，分别计算得分
        scores_total = pd.DataFrame()
        for key, v_list in self.factor_dict.items():
            try:
                dat_tmp = data[v_list]
            except Exception as e:
                print('ddd')
            scores_tmp = get_scores(dat_tmp)
            scores_df = pd.DataFrame({key: scores_tmp})
            scores_total = pd.concat([scores_total, scores_df], axis=1)

        # 按照权重计算得分
        sv = scores_total[self.factor_wei.keys()].values
        wv = np.array([v for v in self.factor_wei.values()])

        scores_sum = pd.Series(index=scores_total.index, data=np.dot(sv, wv))
        # 排序
        scores_sorted = scores_sum.sort_values(ascending=False)

        # todo: 返回的是得分，不是权重，需要进一步修改
        if self.select_type == 'twice_sort':
            basic_scores_df = pd.DataFrame({"basic_scores": scores_sorted})

            # twice_sort_dict = {'factor_n': 'EP',
            #                    'up_or_down': 'down'
            #                    }
            second_f = self.twice_sort_dict['factor_n']  # 'LNCAP_barra'
            di = self.twice_sort_dict['up_or_down']

            value_scores_df = pd.DataFrame({"value_scores": data[second_f]})
            tmp = pd.concat([basic_scores_df, value_scores_df], axis=1)
            # 可以是基本面得分高的前N个中估值低的M个。
            # 也可以是估值高的前N个中基本面得分高的M个。
            # 可以是估值低的前N个中基本面得分高的M个。
            selected_way = {'first': ['basic_scores', 'up', self.first_max],    # 先选择某个因子的最高或是最低的N个股票
                            # 使用BP做为估值因子的话，BP越大估值越低
                            'second': ['value_scores', di, self.max_n]     # 再选择某个因子的最高或是最低的M个股票
                           }
            res = twice_sort(tmp, selected_way)
            res = list(res.index)

        # 选股方式, 分行业选股，每个行业选择5支股票，然后把权重调整为行业权重除以行业内股票数量
        elif self.select_type == 'by_industry':
            scores_sorted = pd.concat([pd.DataFrame({'scores': scores_sorted}), self.industry_map],
                                      axis=1, join='inner')

            res = pd.DataFrame()
            grouped = scores_sorted.groupby('first_industry')
            for key, value in grouped:
                if len(value) > self.one_industry_num:

                    tmp = value.iloc[:self.one_industry_num, :]
                    dd = np.full((self.one_industry_num, 1), constituent_se[key] / self.one_industry_num)
                    tmp_df = pd.DataFrame(data=dd, index=tmp.index, columns=[dt])

                    res = pd.concat([res, tmp_df], axis=0)
                else:
                    dd = np.full((len(value.index), 1), constituent_se[key] / len(value.index))
                    tmp_df = pd.DataFrame(data=dd, index=value.index, columns=[dt])
                    res = pd.concat([res, tmp_df], axis=0)

        elif self.select_type == 'total_num':
            res = list(scores_sorted.index[:self.max_n])

        return res

    # 存储相关结果
    def save(self, pool, save_name='股票池每期结果.csv'):
        basic_save_path = r'D:\pythoncode\IndexEnhancement\股票池'
        save_each_sec_name(pool, save_name)

        # 把最新股票池结果存成txt, 并下载股票池基本信息
        to_save = list(pool.index[pool[pool.columns[-1]] == True])
        add_stock_pool_txt(to_save, '盈利成长选股策略_排名前50', renew=True)
        info = section_stock_infor(to_save)
        info.to_csv(os.path.join(basic_save_path, '股票基本信息.csv'), encoding='gbk')

    # 得到最新股票池
    def latest_pool(self, method, add_infor=False):
        self.get_stock_pool()
        stock_pool = self.rov_of_all_stocks(method)

        if stock_pool.empty:
            return None

        tmp = stock_pool[stock_pool.columns[-1]]
        tmp = tmp[tmp.index[tmp != 0]]
        tmp.sort_values(inplace=True)
        s_list = list(tmp.index)
        # 添加股票名称
        data = Data()
        stock_basic = data.stock_basic_inform
        res = stock_basic.loc[s_list, 'SEC_NAME']
        res_df = pd.DataFrame({'SEC_NAME': res})
        res_df.index.name = 'CODE'

        # res_df.to_csv('D://库存表.csv', encoding='gbk')  TODO

        # 是否添加概念数据和调入股票池时间数据
        if add_infor:
            # 添加调入股票池日期数据
            res_df['跳入股票池日期'] = None
            for k, v in stock_pool.loc[s_list, :].iterrows():
                for i in range(len(v)-1, -1, -1):
                    if not v[i]:
                        res_df.loc[k, '跳入股票池日期'] = v.index[i+1]
                        break

            # 添加概念数据。
            concept = data.concept
            res_df = pd.concat([res_df, concept], axis=1, join='inner')

            for k, v in res_df['CONCEPT'].items():
                try:
                    res_df.loc[k, 'CONCEPT'] = v.replace('[', '').replace(']', '').replace('\'', '')
                except Exception:
                    pass

        return res_df, stock_pool

    def backtest(self, stock_pool, bench='WindA', posi=None, re_invest='re_weighted', hedging=True):

        bt = BackTest(stock_pool, self.freq, adjust_freq=self.freq, position=posi, fee_type='fee',
                      benchmark_str=bench, re_invest=re_invest, hedge_status=hedging)
        bt.run_bt()
        ana = bt.analysis()
        bt.plt_pic()
        return ana

    # 得到特定月份的股票池
    def special_month_pool(self, special_date, stock_pool=None):
        if not isinstance(stock_pool, pd.DataFrame):
            self.get_stock_pool()
            stock_pool = self.main_of_select_stocks_by_scores()

        if stock_pool.empty:
            return None

        ff = None
        for c in stock_pool.columns:
            if c.year == special_date.year and c.month == special_date.month:
                ff = c
                break

        tmp = stock_pool[ff]
        s_list = list(tmp.index[tmp == True])
        return s_list




    # benchmark_r =

        #
        # pool = append_df(stock_pool, target_feq='D')
        #
        # to_del = pool.columns[pool.sum() == 0]
        # pool = pool.drop(to_del, axis=1)
        #
        #
        # pool_wei = append_df(pool_wei, target_feq='D')
        # position = pool_wei

        # 简单回测
        # save_path = r'D:\pythoncode\IndexEnhancement\股票池\计算机'
        # bp = os.path.join(save_path, '计算机_全部等权_日度收益.csv')
        # bp = r'D:\pythoncode\IndexEnhancement\指数相关\WindA.csv'
        # daily_return, net_value, cum_excess_df = easy_bt(position, basic_return_infor=bp)

        # fig_path = os.path.join(save_path, '计算机_全部等权_净值曲线.csv')
        # basic_nv = pd.read_csv(bp, encoding='gbk')
        # basic_nv.index = pd.to_datetime(basic_nv['date'])
        #
        # net_value.index = pd.to_datetime(net_value.index)
        #
        # # net_value
        # dat_plt = pd.DataFrame({'基准走势': basic_nv['net_value'], '净值走势': net_value['net_value']})
        # dat_plt.dropna(how='any', inplace=True)
        # dat_plt = dat_plt / dat_plt.iloc[0, :]
        #
        # fig = plt.figure()
        # plt.plot(dat_plt)
        # plt.legend(dat_plt.columns)
        # plt.show()
        # b_path = r'D:\pythoncode\IndexEnhancement\股票池'
        # # plt.savefig(os.path.join(b_path, "净值走势图.png"))
        #
        # fig = plt.figure()
        # plt.plot(cum_excess_df)
        # plt.show()
        # b_path = r'D:\pythoncode\IndexEnhancement\股票池'
        # plt.savefig(os.path.join(b_path, "超额收益累计走势图.png"))

        # daily_return.to_csv(os.path.join(b_path, '计算机_精选组合_rps及指数择时_日度收益.csv'), encoding='gbk')
        # dat_plt.to_csv(os.path.join(b_path, '盈利成长精选组合_净值曲线.csv'), encoding='gbk')
        # if isinstance(cum_excess_df, pd.DataFrame):
        #     cum_excess_df.to_csv(os.path.join(b_path, '盈利成长精选组合_累计超额收益率.csv'), encoding='gbk')

        # '''
        # return daily_return, net_value, cum_excess_df


# 单独运用的函数，用于每日的跟踪，选出基本面不错的股票池后，再用rps指标来筛一下，然后看看相关的股票
def eps_over_80(stock_pool):
    data = Data()
    rps = data.rps
    tmp = rps[[rps.columns[-1]]]

    stock_pool = pd.concat([stock_pool, tmp], axis=1, join='inner')
    to_save = stock_pool.index[stock_pool[rps.columns[-1]] > 80]
    stock_pool = stock_pool.loc[to_save, :]
    return stock_pool


def growth_stock_pool(method='score', select_tyep='by_industry', bt_or_latest='bt', bm='ZZ500'):
    # 财务绝对要求条件的股票池构建，如roettm大于5%，eps同比增速大于5，sue位于所有股票的top5%。
    financial_dict = {'all': {'scope_0': ('roettm', 3, None),
                              'scope_1': ('basicepsyoy', 5, 500),
                              'scope_2': ('netprofitgrowrate', 5, 500),  # 净利润同比增长率
                              'scope_3': ('debtassetsratio', 1, 60),
                              # 'rise_0': ('netprofitgrowrate', 2)
                              }
                      }

    # indus_dict = {'to_handle_indus': ['有色金属', '钢铁', '采掘', '非银金融'],
    #               # keep 和 delete 两种模式， keep模型下，保留下来上个变量定义的子行业，delete模式下，删除上面的子行业
    #               'handle_type': 'delete',
    #               }

    indus_dict = None
    select_m = select_tyep   # 'total_num'
    # select_m = 'twice_sort'
    twice_sort_dict = {'factor_n': 'Ep',
                       'up_or_down': 'down'
                       }

    factors_wei = {'value': 0.5,
                   'growth': 0.5,
                    }

    risk_factor = {'size': 0}

    para_dict = {'is_enhance': True,
                 'lamda': 10,
                 'turnover': 0.5,
                 'te': None,
                 'industry_max_expose': 0.05,
                 'size_neutral': True,
                 'in_benchmark': False,
                 'max_num': 100,
                }

    pool = FormStockPool(financial_dict, factors_wei, risk_factor=risk_factor, opt_para_dict=para_dict,
                         select_mode=select_m, benchmark=bm, twice_sort_dict=twice_sort_dict, first_max=50,
                         max_n=30, industry_dict=indus_dict)

    if bt_or_latest == 'latest_pool_monthly':
        newest_pool, stock_pool = pool.latest_pool(method=method, add_infor=True)
        newest_pool.to_csv(r'D:\pythoncode\IndexEnhancement\股票池_最终\盈利成长因子选股下个月股票池.csv', encoding='gbk')
        save_path = r'D:\pythoncode\IndexEnhancement\股票池_最终'

        tod = datetime.today()
        per_month = datetime(tod.year, tod.month - 1, 1)
        per_pre_month = datetime(tod.year, tod.month - 2, 1)

        sl = pool.special_month_pool(per_pre_month, stock_pool=stock_pool)
        res1, res2 = month_return_compare_to_market_index(sl, per_month)
        res1.to_csv(os.path.join(save_path, '盈利成长因子上个月组合表现.csv'), encoding='gbk')
        res2.to_csv(os.path.join(save_path, '盈利成长因子上个月个股表现.csv'), encoding='gbk')

    elif bt_or_latest == 'bt':

        res = pool.run_test()

    elif bt_or_latest == 'latest_pool_daily':
        newest_pool = pool.latest_pool(add_infor=True)
        pool_daily = eps_over_80(newest_pool)
        save_path = r'D:\pythoncode\IndexEnhancement\股票池_最终'
        pool_daily.to_csv(os.path.join(save_path, '盈利成长+Rps_日度股票池.csv'), encoding='gbk')


if '__main__' == __name__:

    factors_to_concat = {
        'vol': ['std_12m', 'std_6m', 'std_3m', 'std_1m'],
        'mom': ['M_reverse_180', 'M_reverse_60', 'M_reverse_20'],
        'liq': ['STOQ_Barra', 'STOM_Barra', 'STOA_Barra'],
        'quality': ['ROA_q', 'ROE_q'],
        'value': ['BP', 'EP'],
        'growth': ['Profit_G_q', 'Sales_G_q', 'ROE_G_q'],
        'size': ['LNCAP_barra', 'MIDCAP_barra'],
    }

    select_type = 'total_num'                     # 'total_num' 'twice_sort'  'by_industry'
    bt_or_latest = 'latest_pool_monthly'          # 'bt' 'latest'  'latest_pool_daily'
    method = 'score'                              # 'regress':
    growth_stock_pool(method, select_type, bt_or_latest)











    # select_type = 'by_industry'
    # bt_or_latest = 'bt'
    # growth_stock_pool(select_type, bt_or_latest)

    # 3月底得到的股票池在4月份的表现
    # sl = pool.special_month_pool(datetime(2020, 3, 1))
    # res1, res2 = month_return_compare_to_market_index(sl, datetime(2020, 4, 1))
    # save_path = r'D:\pythoncode\IndexEnhancement\股票池_最终'
    # res1.to_csv(os.path.join(save_path, '盈利成长因子上个月组合表现.csv'), encoding='gbk')

    # net_value.to_csv(r'D:\pythoncode\IndexEnhancement\股票池\Final_net_value.csv', encoding='gbk')
    # daily_return.to_csv(r'D:\pythoncode\IndexEnhancement\股票池\Final_daily_return.csv', encoding='gbk')
    # cum_excess_df.to_csv(r'D:\pythoncode\IndexEnhancement\股票池\Final_cum_excess.csv', encoding='gbk')
    # tmp = pd.read_csv(r'D:\pythoncode\IndexEnhancement\股票池\基本面因子_daily_return_size_down_20.csv', encoding='gbk')
    # tmp.set_index('date', inplace=True)
    # tmp['excess_rate'] = tmp[tmp.columns[0]] - tmp[tmp.columns[1]]
    # tmp['cum_excess_rate'] = (tmp['excess_rate'] + 1).cumprod()
    # tmp.to_csv(r'D:\pythoncode\IndexEnhancement\股票池\市值后20与前20的日收益率差的累计值.csv', encoding='gbk')

    # net_value_no_macro = pd.read_csv(r'D:\无宏观因素的走势.csv', encoding='gbk')
    # net_value_no_macro.set_index('date', inplace=True)

    # net_value.to_csv(r'D:\无宏观因素的走势.csv', encoding='gbk')

    # daily_return, net_value, cum_excess_df
    # net_value.to_csv(r'D:\无宏观因素的走势.csv', encoding='gbk')

    # tmp = pd.DataFrame({'结合宏观因子的选股策略': net_value['net_value'],
    #                     '选股策略': net_value_no_macro['net_value']})
    # tmp.to_csv(r'D:\策略结果对比.csv', encoding='gbk')
    #
    # fig = plt.figure()
    # plt.plot(tmp)
    # # plt.plot(net_value)
    # plt.legend(tmp.columns)
    # plt.show()

    # p = r'D:\pythoncode\IndexEnhancement\股票池\基本面因子_排名前50至前20的结果汇总.csv'
    # tmp = pd.read_csv(p, encoding='gbk')
    # tmp.set_index('date', inplace=True)
    # fig = plt.figure()
    # plt.plot(tmp)
    # # plt.plot(net_value)
    # plt.legend(tmp.columns)
    # plt.show()




