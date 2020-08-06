import os
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from openpyxl import load_workbook
from barra_cne6.barra_template import Data
from utility.relate_to_tushare import generate_months_ends
import pandas.tseries.offsets as toffsets
from utility.factor_data_preprocess import adjust_months, add_to_panels, align

register_matplotlib_converters()
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)  # 图片尺寸设定（宽 * 高 cm^2)

strategy_index_path = r'D:\私募基金相关\私募基金策略指数'


def bool_2_ones(d_df, use_df=False):
    d_ar = d_df.values
    tmp_a = np.full(d_ar.shape, 0)
    tmp_a[d_ar == True] = 1

    if use_df:
        res = pd.DataFrame(data=tmp_a, index=d_df.index, columns=d_df.columns)
    else:
        res = tmp_a
    return res


# 把私募产品数据的列名设置为 日期、净值，注意频率，一般为周频数据


class PrivateEquity:
    def __init__(self, inputs, frequency, benchmark, rf_rate=0.04):
        # ex_ret 是否需要计算超额收益率
        # self.freq 可以是 'y' 'q' 'm' 'w' 'd'
        self.portfolio = None  # 组合净值每日记录
        self.freq = frequency
        self.rf_rate = rf_rate
        self.benchmark = None
        self.benchmark_name = None
        if isinstance(inputs, str):
            self.load_data_str(inputs)
        elif isinstance(inputs, pd.DataFrame):
            self.load_data_df(inputs)

        self.load_benchmark(benchmark)
        # self.benchmark = benchmark

    def load_data_str(self, path):
        ext = path.split('.')[-1]
        if ext == 'csv':
            dat = pd.read_csv(path, encoding='gbk', engine='python')
        elif ext == 'xlsx':
            dat = pd.read_excel(path, encoding='gbk')
        else:
            msg = f"不支持的文件存储类型：{ext}"
            raise TypeError(msg)

        if '日期' not in self.portfolio.columns or '净值' not in self.portfolio.columns:
            print('日期或净值命名错误')
            raise ValueError

        self.portfolio = dat
        self.portfolio.set_index('日期', inplace=True)
        self.trim_df()

    def load_data_df(self, dat):
        if len(dat.columns) == 2 and '日期' not in dat.columns:
            dat.columns = ['日期', '净值']

        self.portfolio = dat

        if '日期' in self.portfolio.columns:
            self.portfolio.set_index('日期', inplace=True)
        if isinstance(self.portfolio.index[0], (int, np.int64)):
            new_index = []
            for d in self.portfolio.index:
                new_index.append(datetime.strptime(str(d), "%Y%m%d"))

            self.portfolio.index = new_index

        self.portfolio['净值'] = self.portfolio['净值'].apply(lambda x: float(x) if isinstance(x, str) else x)

        self.trim_df()

    # 把净值调整为从1开始
    def trim_df(self):
        if self.portfolio.loc[self.portfolio.index[0], '净值'] != 1.0:
            self.portfolio['净值'] = self.portfolio['净值']/self.portfolio.loc[self.portfolio.index[0], '净值']

    def load_benchmark(self, benchmark_type):

        if '净值' not in benchmark_type.columns:
            if '收盘价' in benchmark_type.columns:
                benchmark_type = benchmark_type.rename({'收盘价': '净值'}, axis=1)
            elif '收盘价(元)' in benchmark_type.columns:
                benchmark_type = benchmark_type.rename({'收盘价(元)': '净值'}, axis=1)
            else:
                print('debug，沒有找到净值且未找到相对应的替换项')
                raise ValueError

        if isinstance(benchmark_type, pd.DataFrame):
            self.benchmark = benchmark_type
            self.benchmark_name = '基准'
        else:
            print('无基准收益')
            return None

        # 基准数据比产品净值数据短，把产品净值数据截断，提取前期的数据。
        # 把前面的截断
        if self.benchmark.index[0] > self.portfolio.index[0]:
            to_del = []
            for d in self.portfolio.index:
                if not (d.year == self.benchmark.index[0].year and d.month == self.benchmark.index[0].month):
                    to_del.append(d)
                else:
                    break

            self.portfolio.drop(to_del, axis=0, inplace=True)

        # 把后面的截断
        if self.benchmark.index[-1] < self.portfolio.index[-1]:
            to_del = []
            for d in range(len(self.portfolio.index) - 1, -1, -1):
                if self.benchmark.index[-1] < self.portfolio.index[d]:
                    to_del.append(self.portfolio.index[d])
                else:
                    break

            self.portfolio.drop(to_del, axis=0, inplace=True)

        self.trim_df()

        # 处理基准的数据, is_inner的意思，是基准数据的日期与策略净值数据的日期相同，切基准的日期包含了净值数据的日期。
        new_index = [i for i in self.benchmark.index if i in self.portfolio.index]
        if len(new_index) == len(self.portfolio.index):
            self.benchmark = self.benchmark.loc[new_index, :]
            self.benchmark['净值'] = self.benchmark['净值'] / self.benchmark.loc[self.benchmark.index[0], '净值']
        else:
            start = self.portfolio.index[0]
            end = self.portfolio.index[-1]
            st_loc = np.argmin(np.abs(self.benchmark.index - start))
            ed_loc = np.argmin(np.abs(self.benchmark.index - end))
            self.benchmark = self.benchmark.iloc[st_loc: ed_loc, :]

    def summary(self):

        bench_return = self._bench_return()  # 基准对应区间的累计收益率
        ann_ret = self._annual_return(None)  # 年化收益
        ann_vol = self._annual_vol(None)  # 年化波动
        max_wd = self._max_drawdown(None)  # 最大回撤
        sharpe = self._sharpe_ratio(ann_ret=ann_ret, ann_vol=ann_vol)  # 夏普比率
        ann_excess_ret = self._ann_excess_ret()    # 年化超额收益
        # ic_rate = self._ic_rate(start_date, end_date)

        recommend = None
        if not ann_excess_ret:
            recommend = '无基准，暂无推荐'
        elif ann_ret > 0 and ann_excess_ret > 0:
            recommend = '建议客户继续持有或追加投资'
        elif ann_ret > 0 > ann_excess_ret > -0.05:
            recommend = '建议客户继续持有'
        elif ann_ret > 0 and ann_excess_ret < -0.05:
            recommend = '建议客户减持'
        elif ann_ret < 0 < ann_excess_ret:
            recommend = '建议客户继续持'
        elif -0.1 < ann_ret < 0 and ann_excess_ret < 0:
            recommend = '建议客户减持'
        elif ann_ret < -0.10 and ann_excess_ret < 0:
            recommend = '建议客户全部赎回'

        try:
            self.portfolio.iloc[-1, 0] / self.portfolio.iloc[0, 0] - 1,
        except Exception as e:
            print('debug')

        summary = {
            '开始日期': self.portfolio.index[0].strftime("%Y-%m-%d"),
            '截至日期': self.portfolio.index[-1].strftime("%Y-%m-%d"),
            '累计收益': self.portfolio.iloc[-1, 0] / self.portfolio.iloc[0, 0] - 1,
            '基准名称': self.benchmark_name,
            '基准对应区间累计收益': bench_return,
            '年度收益': ann_ret,
            '年度波动': ann_vol,
            '最大回撤': max_wd,
            '夏普比率': sharpe,
            '年化超额收益': ann_excess_ret,
            '建议': recommend,
        }
        return pd.Series(summary)

    def return_each_year(self):
        st_y = self.portfolio.index[0].year
        ed_y = self.portfolio.index[-1].year

        res = pd.DataFrame(index=range(st_y, ed_y + 1, 1), columns=['年度收益', '最大回撤'])
        for y in range(st_y, ed_y + 1, 1):
            tmp = self.portfolio.loc[self.portfolio.index[self.portfolio.index.year == y], :]
            tmp_start = tmp.index[0]
            tmp_end = tmp.index[-1]
            max_wd = self._max_drawdown(acc_rets=None, start_date=tmp_start, end_date=tmp_end)    # 最大回撤

            res.loc[y, '年度收益'] = tmp.loc[tmp.index[-1], '净值'] / tmp.loc[tmp.index[0], '净值'] - 1
            res.loc[y, '最大回撤'] = max_wd
            if isinstance(self.benchmark, pd.DataFrame):
                if tmp_start in self.benchmark.index and tmp_end in self.benchmark.index:
                    res.loc[y, '基准收益'] = self.benchmark.loc[tmp_end, '净值'] / \
                                                     self.benchmark.loc[tmp_start, '净值'] - 1

        res.index.name = '年份'
        return res

    def plot_pic(self, save_name):

        tmp = pd.DataFrame({'产品净值': self.portfolio['净值'], '基准净值': self.benchmark['净值']})
        tmp.dropna(how='any', inplace=True)
        plt.plot(tmp)
        # plt.title(fname)
        plt.legend(tmp.columns, loc=0)
        plt.savefig(os.path.join(r'D:\私募基金相关\pic', save_name + f'.png'))
        plt.close()

        return None

    def _get_date_gap(self, freq='d'):

        start_date = self.portfolio.index[0]
        end_date = self.portfolio.index[-1]
        days = (end_date - start_date) / toffsets.timedelta(1)

        return days

    def _annual_return(self, net_vals=None):

        if net_vals is None:
            net_vals = self.portfolio['净值']
            start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        else:
            start_date, end_date = net_vals.index[0], net_vals.index[-1]

        tmp = list(net_vals.index)
        start_date_id = tmp.index(start_date)
        end_date_id = tmp.index(end_date)
        if start_date_id == 0:
            net_vals = net_vals[start_date_id:end_date_id + 1]
            # 净值从1开始
            total_ret = net_vals.values[-1] / net_vals.values[0] - 1
        else:
            # 对于月频数据，计算收益从start_date_id前一个
            net_vals = net_vals[start_date_id - 1:end_date_id + 1]
            total_ret = net_vals.values[-1] / net_vals.values[0] - 1

        date_gap = self._get_date_gap(freq=self.freq)
        exp = 365 / date_gap
        ann_ret = (1 + total_ret) ** exp - 1

        return ann_ret

    def _annual_vol(self, net_vals=None):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if 'netval_pctchg' not in self.portfolio.columns:
            self.portfolio['netval_pctchg'] = self.portfolio['净值'] / self.portfolio['净值'].shift(1)

        if net_vals is None:
            ret_per_period = self.portfolio['netval_pctchg']

        ret_per_period = ret_per_period.loc[start_date:end_date]
        # 年化波动率 = 日频收益率标准差 * sqrt(250)
        # 年化波动率 = 周频收益率标准差 * sqrt(52)
        # 年化波动率 = 月频收益率标准差 * sqrt(12)
        ret_per_period = ret_per_period.dropna()
        if self.freq == 'y':
            ann_vol = ret_per_period.std()
        elif self.freq == 'q':
            ann_vol = ret_per_period.std() * np.sqrt(4)
        elif self.freq == 'm':
            ann_vol = ret_per_period.std() * np.sqrt(12)
        elif self.freq == 'w':
            ann_vol = ret_per_period.std() * np.sqrt(52)
        elif self.freq == 'd':
            ann_vol = ret_per_period.std() * np.sqrt(250)

        return ann_vol

    def _max_drawdown(self, acc_rets=None, start_date=None, end_date=None):

        if not start_date and not end_date:
            start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]

        if acc_rets is None:
            acc_rets = self.portfolio['净值'] - 1

        acc_rets = acc_rets.loc[start_date:end_date]
        max_drawdown = (1 - (1 + acc_rets) / (1 + acc_rets.expanding().max())).max()
        return max_drawdown

    def _sharpe_ratio(self, ann_ret=None, ann_vol=None):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if ann_ret is None:
            ann_ret = self._annual_return(start_date, end_date)
        if ann_vol is None:
            ann_vol = self._annual_vol(start_date, end_date)
        return (ann_ret - self.rf_rate) / ann_vol

    def form_daily_return(self, net_value):
        # net_value = self.portfolio
        res = net_value['净值'] / net_value['净值'].shift(1) - 1
        res.dropna(inplace=True)
        return res

    def _get_excess_acc_ret(self):

        if not isinstance(self.benchmark, pd.DataFrame):
            return None

        dr_s = self.form_daily_return(self.portfolio)
        dr_b = self.form_daily_return(self.benchmark)

        dr_e = dr_s - dr_b
        res = (dr_e+1).cumprod()

        return res

    def _ann_excess_ret(self):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if isinstance(self.benchmark, pd.DataFrame):
            if len(self.benchmark.index) == len(self.portfolio.index):
                excess_acc_ret = self._get_excess_acc_ret()
                ann_excess_ret = self._annual_return(net_vals=excess_acc_ret)
                return ann_excess_ret
            else:
                bench_return = self.benchmark.loc[self.benchmark.index[-1], '净值'] / \
                               self.benchmark.loc[self.benchmark.index[0], '净值'] \
                               - 1
                portfolio_return = self.portfolio.loc[self.portfolio.index[-1], '净值'] / \
                                   self.portfolio.loc[self.portfolio.index[0], '净值'] \
                                   - 1
                ann_excess_ret = portfolio_return - bench_return

                days = np.min([(self.portfolio.index[-1] - self.portfolio.index[0]).days,
                              (self.benchmark.index[-1] - self.benchmark.index[0]).days])

                exp = 365 / days
                ann_excess_ret = (1 + ann_excess_ret) ** exp - 1

                return ann_excess_ret
        else:
            return None

    # def _get_excess_acc_ret(self, start_date=None, end_date=None):
    #
    #     bm_ret = self.portfolio_record['benchmark']
    #     nv_ret = self.portfolio_record['netval_pctchg']
    #
    #     if start_date and end_date:
    #         bm_ret = bm_ret.loc[start_date:end_date]
    #         nv_ret = nv_ret.loc[start_date:end_date]
    #     excess_ret = nv_ret.values.flatten() - bm_ret.values.flatten()
    #     excess_acc_ret = pd.Series(np.cumprod(1+excess_ret), index=nv_ret.index)
    #     return excess_acc_ret

    def _winning_rate(self):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        nv_pctchg = self.portfolio['netval_pctchg']

        if start_date and end_date:
            nv_pctchg = nv_pctchg.loc[start_date:end_date]

        nv_pctchg = nv_pctchg.dropna()
        win = (nv_pctchg > 1)
        win_rate = np.sum(win) / len(win)
        return win_rate

    # def _winning_rate_over_benchmark(self, start_date=None, end_date=None):
    #     if self.benchmark:
    #         nv_pctchg = self.portfolio['netval_pctchg']
    #         bm_pctchg = self.portfolio['benchmark']
    #
    #         if start_date and end_date:
    #             nv_pctchg = nv_pctchg.loc[start_date:end_date]
    #             bm_pctchg = bm_pctchg.loc[start_date:end_date]
    #
    #         win_daily = (nv_pctchg > bm_pctchg)
    #         win_rate = np.sum(win_daily) / len(win_daily)
    #         return win_rate
    #     else:
    #         return None

    def _bench_return(self):  # 基准对应区间的累计收益率

        if not isinstance(self.benchmark, pd.DataFrame):
            return None

        bench_return = self.benchmark.loc[self.benchmark.index[-1], '净值'] / self.benchmark.loc[self.benchmark.index[0], '净值'] \
                       - 1

        return bench_return


# 回测类：逻辑上仅处理月频调仓的处理，对于需要用到日频率数据的freq设为D，不需要的freq设为M，暂不用涉及到季度调仓的策略。
class BackTest:
    def __init__(self, wei, freq, adjust_freq='M', fee_type='fee', benchmark_str='WindA',
                 hedge_status=True, hedge_para_dict={}):
        self.weight = wei                        # 股票权重
        self.freq = freq                         # 处理类型：频率
        self.adjust_freq = adjust_freq           # 调仓评论
        self.adjust_days = None                  # 调仓日
        self.changePCT_np = None                 # 价格变动百分比
        self.net_value = None
        self.net_pct = None
        self.fee_type = fee_type
        self.load_pct()
        self.benchmark_p = None                  # 基准指数的表现
        self.benchmark_r = None
        self.load_benchmark(benchmark_str)

        self.summary = None                      # 策略评价指标
        self.hedging = hedge_status
        self.hedge_para_dict = hedge_para_dict   # 做对冲策略时的一些参数

    def load_benchmark(self, benchmark_str):
        data = Data()
        if self.freq == 'M':
            if benchmark_str in ['WindA', 'HS300', 'SH50', 'ZZ500']:
                price_monthly = data.index_price_monthly
                self.benchmark_p = price_monthly.loc[benchmark_str, :]
                self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1
            else:
                price_monthly = data.industry_price_monthly
                self.benchmark_p = price_monthly.loc[benchmark_str + '（申万）', :]
                self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1

    # 自然载入变动百分比数据
    def load_pct(self):
        if self.freq == 'M':         # 导入月度价格变动百分比数据
            data = Data()
            self.changePCT_np = data.changepct_monthly.shift(-1, axis=1)
        if self.freq == 'D':
            print('未实现')

    def run_bt(self):
        if self.freq == 'M':   # 月频数据、月频调仓，权重与价格百分比直接相乘
            self.month_bt()
        if self.freq == 'D':
            print('未实现')

    # 程序逻辑： 对于月频率数据，如果月频调仓，那么就比较简单，知道权重和change相乘就可以。
    def month_bt(self):

        # 先对行和列取交集，然后再转换为array，再矩阵乘法得到每个月的收益，再判断是不是要去费，
        [self.weight, self.changePCT_np] = align(self.weight, self.changePCT_np)
        self.weight.fillna(0, inplace=True)
        self.changePCT_np.fillna(0, inplace=True)

        # 净值数据
        if self.fee_type == 'No_fee':
            ret_df = self.weight * self.changePCT_np
            # 月度收益pct
            self.net_pct = ret_df.sum(axis=0)
            self.net_value = (self.net_pct + 1).cumprod()
        else:
            # 计算出，每个月仓位变动中，没有变动的比例和变动的比率是多少
            unchanged_wei = pd.DataFrame(0, index=self.weight.index, columns=self.weight.columns)
            buy_wei = pd.DataFrame(0, index=self.weight.index, columns=self.weight.columns)
            sell_wei = pd.DataFrame(0, index=self.weight.index, columns=self.weight.columns)
            for col_i in range(0, len(self.weight.columns)):
                if col_i == 0:
                    buy_wei[self.weight.columns[col_i]] = self.weight[self.weight.columns[col_i]]
                else:
                    pre_i = col_i - 1
                    unchanged_wei[self.weight.columns[col_i]] = self.weight[self.weight.columns[pre_i:col_i + 1]].min(
                        axis=1)

                    cha = self.weight[self.weight.columns[col_i]] - self.weight[self.weight.columns[pre_i]]
                    buy_wei[self.weight.columns[col_i]] = cha.where(cha > 0, 0)
                    sell_wei[self.weight.columns[col_i]] = abs(cha.where(cha < 0, 0))

            wei_ar = self.weight.values
            change_pct_ar = self.changePCT_np.values
            unchanged_wei_ar = unchanged_wei.values
            buy_wei_ar = buy_wei.values
            sell_wei_ar = sell_wei.values

            # 双边佣金率0.02%，建仓冲击成本0.5%，减仓冲击成本0.3%，卖出单边印花税0.1%
            tax = 0.0001
            fee = 0.00002
            buy_impact_cost = 0.001
            sell_impact_cost = 0.001

            net_value_ar = np.zeros(wei_ar.shape)
            h_n, l_n = net_value_ar.shape
            net_value_se = np.zeros(l_n)
            for col in range(0, l_n):
                wei_tmp = wei_ar[:, col]
                change_pct_tmp = change_pct_ar[:, col]
                unchanged_wei_tmp = unchanged_wei_ar[:, col]
                buy_wei_tmp = buy_wei_ar[:, col]
                sell_wei_tmp = sell_wei_ar[:, col]

                if col != 0:

                    sell_wei_tmp.sum()
                    net_value_ar[:, col - 1].sum()
                    # 先减仓后才可加仓，建仓时产生的冲击成本对整体净值有影响
                    fee1 = (net_value_ar[:, col - 1] * sell_wei_tmp).sum() * (1 - sell_impact_cost) * tax   # 交的印花税
                    fee2 = (net_value_ar[:, col - 1] * sell_wei_tmp * sell_impact_cost).sum()            # 卖出冲击成本导致的减值
                    total_sell_impact_cost = fee1 + fee2

                # 新建仓的股票的本期收益，新建仓时的冲击成本对本期收益产生影响
                new_set = buy_wei_tmp * (1-fee) * (1 + change_pct_tmp/(1 + buy_impact_cost))
                # 仓位不变的股票的本期收益，仓位不变，不收冲击成本影响
                unc = unchanged_wei_tmp * (1 + change_pct_tmp)

                # 上期净值先减去卖出的成本得到本期基准净值，然后再得到本期的收益净值
                if col != 0:
                    net_value_ar[:, col] = net_value_se[col-1] * (1-total_sell_impact_cost) * (new_set + unc)
                    net_value_se[col] = net_value_ar[col].sum()
                else:
                    net_value_ar[:, col] = new_set
                    net_value_se[col] = new_set.sum()

            net_value = pd.Series(net_value_se, index=self.weight.index)

            return net_value

    def hedging_fun(self):
        pass




    def analysis(self):
        dat_df = pd.DataFrame({'净值': self.net_value})
        dat_df.index.name = '日期'
        bm_df = pd.DataFrame({'净值': self.benchmark_p})
        ana = PrivateEquity(dat_df, self.freq.lower(), benchmark=bm_df)
        summary = ana.summary()
        summary = pd.DataFrame({'评价指标': summary})
        self.summary = summary
        return summary

    def plt_pic(self):
        pic_df = pd.DataFrame({'net_value': self.net_value, 'basic': self.benchmark_p})
        pic_df.dropna(how='any', inplace=True)
        pic_df = pic_df/pic_df.iloc[0, :]
        fig = plt.figure()
        plt.plot(pic_df)
        plt.legend(pic_df.columns)
        plt.show()


if __name__ == "__main__":
    stock_wt = pd.read_csv(r'D:\test.csv', encoding='gbk')
    stock_wt.set_index(stock_wt.columns[0], inplace=True)
    stock_wt.columns = pd.to_datetime(stock_wt.columns)

    bt = BackTest(stock_wt, 'M', benchmark_str='HS300')
    bt.run_bt()
    bt.plt_pic()
    print(bt.analysis())


