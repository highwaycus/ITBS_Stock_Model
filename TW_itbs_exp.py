import os
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import datetime
import time
import random
import requests
import html5lib
# from selenium import webdriver
# from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import multiprocessing as mp

from production_setting import sub_process_bar, tw_path_setting, tw_price_df_loading, display_setting

'''
If Investemnt Trust have no action for a long time and suddenly have a new action, we will track on it.
Some industries have special features and cannot use this strategy.
You cannot keep crawling... 
'''

display_setting()


def itbs_exp_dir_setting():
    base_dir = tw_path_setting(collapse='daily')[0][:-9]
    if 'AI_Public_Data' in base_dir:
        base_dir += 'Stock_KChart/'
    exp_save_dir = base_dir + 'exp_tmp/'
    return exp_save_dir


def crawling_goodinfo(ticker='8255'):
    # https://bradnopitt.blogspot.com/2018/02/pandas.html
    url = 'https://goodinfo.tw/StockInfo/ShowK_Chart.asp?STOCK_ID={}&CHT_CAT2=DATE'.format(ticker)
    # url='https://goodinfo.tw/StockInfo/ShowBuySaleChart.asp?STOCK_ID=2104&CHT_CAT=DATE'
    resp = requests.get(url, headers={
        'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like gecko) Chrome/63.0.3239.132 Safari/537.36'})
    time.sleep(random.randrange(1, 4))
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    time.sleep(random.randrange(1, 3))
    table = soup.find(id="divK_ChartDetail")
    # table = soup.find(id="divBuySaleDetailData")
    # for i in range(1, len(table.text)):
    #     if (table.text[i-11:i] == '賣出(張)買賣超(張)') and (table.text[i+3] == '/'):
    #         break

    for i in range(1, len(table.text)):
        if table.text[i - 8: i + 1] == '餘額 增減 餘額 ':
            break
    i += 1
    data = table.text[i:]
    record = pd.DataFrame(columns=['date', '證券代號', '證券名稱', '買進股數', '賣出股數', '買賣超股數'])
    len_text = len(data)
    i = 0
    today_year = datetime.datetime.today().year
    while i < len_text:
        if data[i] == '/':
            date_i = data[i - 2:i + 3]
            date_i = int(str(today_year) + date_i.replace('/', ''))
            j = i + 4
            while (j < len(data)) and (data[j] != '/'):
                j += 1
            row_ = data[i + 4:j].split(' ')
            net = row_[12]
            if not len(net):
                net = 0
            else:
                net = int(net.replace(',', ''))
            record.loc[i] = [date_i, ticker, '', np.NAN, np.NAN, net]
        i += 1
    record = record.sort_values(by=['date']).reset_index(drop=True)
    if not len(record):
        print('No data for {} from GoodInfo'.format(ticker))
    else:
        print('Crawling {} from GoodInfo'.format(ticker))
    return record


def ticker_price_loading_combine(ticker, price_path, tmp_df, h_list=[1, 2, 3, 5, 10]):
    try:
        price = tw_price_df_loading(data_path=price_path, ticker=str(ticker))
    except:
        print('{} has no price data'.format(ticker))
        price = None
    if (price is None) or (not len(price)):
        print('{} has no price data'.format(ticker))
        price = None
    if price is None:
        df = tmp_df
    else:
        price_col = ['開盤價', '收盤價', '隔日開盤價', '總市值(億)', '成交量', '成交量變動(%)'] + ['h{}_pct'.format(h) for h in h_list]
        price = price.rename(columns={'日期': 'date'})
        price = price[(price['date'] >= min(tmp_df))]
        price = price.set_index('date', drop=True)
        if '收盤價' in price:
            price['隔日開盤價'] = price['開盤價'].shift(-1)
            for h in h_list:
                price['tmp收盤價_h{}'.format(h)] = price['收盤價'].shift(-h)
                price['h{}_pct'.format(h)] = price.apply(
                    lambda x: x['tmp收盤價_h{}'.format(h)] / x['隔日開盤價'] - 1 if str(x['隔日開盤價']) != 'nan' else np.NAN,
                    axis=1)
                price = price.drop(columns=['tmp收盤價_h{}'.format(h)], axis=1)
            for d in tmp_df:
                if d not in price.index:
                    continue
                for col in price_col:
                    tmp_df[d][col] = price.loc[d, col]
    return tmp_df


########################################################################################################################
class ITBS:
    def __init__(self, start=20150101, end=None, silence_days=25, max_p=2000, min_p=100000, max_mktcap=100, mode='backtest',
                 exp_save_dir=None, all_record=None, h_list=[1, 2, 3, 5, 10]):
        """
        :param start:int, YYYYmmdd
        :param self.all_record: dict, {[int(YYYYnndd)]:{'columns':[col_name1, ..., colname5], ticker1:[col1, col2,...,col5], ticker2:[col1, col2,...,col5]...}}
        """
        if exp_save_dir is None:
            self.exp_save_dir = itbs_exp_dir_setting()
        try:
            os.mkdir(self.exp_save_dir)
        except FileExistsError:
            pass
        self.start = start
        self.start_yr = int(str(self.start)[:4])
        self.silence_days = silence_days
        self.max_p = max_p
        self.min_p=min_p
        self.mode=mode
        self.max_mktcap = max_mktcap
        self.signal_item = 'signal_si-{}d-max{}-min{}'.format(silence_days, max_p, min_p)
        self.h_list = h_list
        if end is None:
            self.end = int(datetime.datetime.today().strftime('%Y%m%d'))
        else:
            self.end = end
        self.end_yr = int(str(self.end)[:4])
        try:
            self.all_record = np.load(
                tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record.npy',
                allow_pickle=True).item()
        except FileNotFoundError:
            self.all_record = {}
        try:
           self.trans_record = np.load(
                tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                allow_pickle=True).item()
        except:
           self.trans_record = {}
        self.ticker_itbs_format = '{}_it_bet_bs_record.npy'

    def data_loading(self):
        try:
            self.all_record = np.load(
                tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record.npy',
                allow_pickle=True).item()
        except FileNotFoundError:
            self.all_record = {}
        if len(self.all_record) > 0:
            start = datetime.datetime.strptime(str(max(self.all_record)), '%Y%m%d') + datetime.timedelta(days=1)
        else:
            start = datetime.datetime.strptime(str(self.start), '%Y%m%d')
        # today = datetime.datetime.today()
        today = datetime.datetime.combine(datetime.datetime.today(), datetime.datetime.min.time())
        date_datetime = start

        delta = today - date_datetime
        jj, total_days = 1, delta.days + 1
        if (total_days == 1)  and (max(self.all_record) == int(date_datetime.strftime('%Y%m%d'))):
            print('No need to update ITBS')
            return
        update_date_list = []
        while date_datetime <= datetime.datetime.combine(today, datetime.datetime.min.time()):
            sleep_t = random.randrange(15, 18)
            time.sleep(sleep_t)
            date = int(date_datetime.strftime('%Y%m%d'))
            url = 'https://www.twse.com.tw/fund/TWT44U?response=json&date={}_='.format(date)
            resp = requests.get(url)
            if date < 20220126:
                try:
                    soup = BeautifulSoup(resp.text, 'lxml')
                    if soup.body is None:
                        pass
                    else:
                        body_text = soup.body.p.text
                except:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    body_text = soup.text
            else:
                soup = BeautifulSoup(resp.text, 'html.parser')
                try:
                    body_text = soup.body.p.text
                except:
                    body_text = soup.text
                    # break

                    # http: // fubon - ebrokerdj.fbs.com.tw / Z / ZG / ZGK_DD.djhtm
                    # https: // www.tpex.org.tw / web / stock / 3insti / daily_trade / 3itrade_hedge.php?l = zh - tw
            record = []
            time.sleep(3)
            if True:
                if '查詢日期小於93年12月17日，請重新查詢' in body_text:
                    print('Error:重新查詢')
                    time.sleep(20)
                    url = 'https://www.twse.com.tw/fund/TWT44U?response=json&date={}_='.format(date)
                    '''
                    New version
                    https://www.twse.com.tw/zh/page/trading/fund/T86.html
                    '''
                    resp = requests.get(url)
                    soup = BeautifulSoup(resp.text, 'lxml')
                    body_text = soup.body.p.text
                elif '沒有符合條件的資料' in body_text:
                    pass
                else:
                    full_len = len(body_text)
                    i = 5
                    while (body_text[i - 6:i] != '"data"') and (i < full_len):
                        i += 1
                    j = i + 2  # body_text[j] = '['
                    while (j < len(body_text) - 1) and (body_text[j:j + 2] != ']]'):
                        if body_text[j] == '[':
                            k = j + 1
                            while body_text[k] != ']':
                                k += 1
                            new_item = body_text[j + 1: k].split('","')
                            record.append(new_item)
                        j += 1
                    i = 7
                    while (body_text[i - 8:i] != '"fields"') and (i < full_len):
                        i += 1
                    ii = i + 2
                    while body_text[ii] != ']':
                        ii += 1
                    column_text = body_text[i + 2:ii].split('","')
                    column_text = [co.replace('"', '') for co in column_text]
                    record_dict_i = {}
                    record_dict_i['columns'] = column_text[1:]
                    for c in record:
                        c[1] = c[1].replace(' ', '')
                        c[2] = c[2].replace(' ', '')
                        for k in range(3, 6):
                            c[k] = int(c[k].replace(',', '').replace('"', ''))
                        c = c[1:]
                        record_dict_i[c[0]] = c
                    self.all_record[date] = record_dict_i
                    update_date_list.append(date)
            jj = sub_process_bar(jj, total_days)
            date_datetime += datetime.timedelta(days=1)
            np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record.npy',
                    self.all_record,
                    allow_pickle=True)
        self.bench_last_date = max(self.all_record)
        self.update_date_list = update_date_list
        np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record.npy', self.all_record,
                allow_pickle=True)

    def itntbs_record_transform(self, save=True):
        """
        :param self.all_record: dict
        :param save: bool, default True
        :return: dict, key = str(ticker_id), {ticker1:{col1:{}....}}
        """
        try:
           self.trans_record = np.load(
                tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                allow_pickle=True).item()
        except:
           self.trans_record = {}
        if self.all_record is None:
            try:
                self.all_record = np.load(
                    tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record.npy',
                    allow_pickle=True).item()
            except FileNotFoundError:
                self.all_record = {}

        if not hasattr(self, 'update_date_list'):
            self.update_date_list = list(self.all_record.keys())
        jj, total_len = 1, len(self.update_date_list)
        for d in self.update_date_list:
            jj = sub_process_bar(jj, total_len)
            col_list = self.all_record[d]['columns'].copy()
            for ticker in self.all_record[d]:
                if ticker == 'columns':
                    continue
                if ticker not in self.trans_record:
                    # print('{} not in self.trans_record, create...'.format(ticker))
                    self.trans_record[ticker] = {d: {c:np.NAN for c in col_list}}
                if d not in self.trans_record[ticker]:
                    self.trans_record[ticker][d] = {c:np.NAN for c in col_list}
                for i in range(len(col_list)):
                    self.trans_record[ticker][d][col_list[i]] = self.all_record[d][ticker][i]
        # Fill in no data date
        print('\nFilling in No Data Date...')
        jj, total_len = 1, len(self.trans_record)
        for ticker in self.trans_record:
            jj = sub_process_bar(jj, total_len)
            loss_date = [d for d in self.all_record if d not in self.trans_record[ticker]]
            if len(loss_date):
                name = self.trans_record[ticker][max(self.trans_record[ticker])]['證券名稱']
            else:
                continue
            for di in loss_date:
                self.trans_record[ticker][di] = {}
                for i in range(len(col_list)):
                    if col_list[i] == '證券名稱':
                        self.trans_record[ticker][di][col_list[i]] = name
                    elif col_list[i] == '證券代號':
                        self.trans_record[ticker][di][col_list[i]] = ticker
                    else:
                        self.trans_record[ticker][di][col_list[i]] = 0
        if save:
            np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                   self.trans_record, allow_pickle=True)

    def shrink_trans_record(self):
        jj, total_len = 1, len(self.trans_record)
        for ticker in self.trans_record:
            jj = sub_process_bar(jj, total_len)
            if type(self.trans_record[ticker]) is dict:
                continue
            # print(ticker)
            vv =self.trans_record[ticker].copy()
            vv = {c: vv[c] for c in vv}
            self.trans_record[ticker] = vv

    def supplement_from_goodinfo(self):
        for file in os.listdir(tw_path_setting(collapse='daily')[0]):
            if not file.startswith('日收盤'):
                continue
            ticker = file.split('_')[1].replace('.csv', '')
            # if (ticker not in self.trans_record) or (not len(trans_record[ticker])) or (not len(trans_record[ticker]['date'])):
            if ticker not in self.trans_record:
                time.sleep(random.randrange(16, 20))
                try:
                    ticker_record = crawling_goodinfo(ticker=ticker)
                    ticker_record = {c: ticker_record[c].tolist() for c in ticker_record}
                    self.trans_record[ticker] = ticker_record
                    np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                           self.trans_record, allow_pickle=True)
                except:
                    print('Crawling {} from GoodInfo FAIL'.format(ticker))
        np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                self.trans_record, allow_pickle=True)

    def itntbs_record_trasform_main(self, save=True, crawling=False):
        self.itntbs_record_trasform(self.all_record, save=save)
        if crawling:
            self.trans_record = self.supplement_from_goodinfo()
        self.shrink_trans_record()
        np.save(tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',self.trans_record,
                allow_pickle=True)

    def it_bet_bs_job(self, input_):
        ticker, price_path, tmp_df = input_[0], input_[1], input_[2]
        if not len(tmp_df):
            print('{} has no data in self.trans_record'.format(ticker))
            return
        if self.mode == 'backtest':
            tmp_dff = ticker_price_loading_combine(ticker, price_path, tmp_df, self.h_list)
        elif self.mode == 'exp':
            if os.path.isfile(self.exp_save_dir + self.ticker_itbs_format.format(ticker)):
                tmp_df = np.load(self.exp_save_dir + self.ticker_itbs_format.format(ticker), allow_pickle=True).item()
            else:
                tmp_df = ticker_price_loading_combine(ticker, price_path, tmp_df, self.h_list)
        if self.mode == 'exp':
            if max(tmp_df) == self.bench_last_date:
                if self.signal_item in tmp_df[max(tmp_df)]:
                    return
        df = pd.DataFrame(tmp_df).transpose().sort_index()
        df['accumulation_{}days_net'.format(self.silence_days)] = df['買賣超股數'].rolling(window=self.silence_days).sum().shift(1)
        if '收盤價' in df:
            df['pct_{}d'.format(self.silence_days)] = df['收盤價'].pct_change(self.silence_days)
        df[self.signal_item] = df.apply(
            lambda x: 1 if (abs(x['accumulation_{}days_net'.format(self.silence_days)]) < self.max_p) and (x['買賣超股數'] > self.min_p) and (
                    str(x['accumulation_{}days_net'.format(self.silence_days)]) != 'nan') else 0, axis=1)
        if '總市值(億)' in df:
            df['總市值(億)'] = df['總市值(億)'].fillna(method='ffill')
            df[self.signal_item + '_mktcap{}'.format(self.max_mktcap)] = df.apply(
                lambda x: x[self.signal_item] if x['總市值(億)'] < self.max_mktcap else 0, axis=1)
        tmp_df = {d:{c: df.loc[d, c] for c in df.columns} for d in df.index}
        np.save(self.exp_save_dir + self.ticker_itbs_format.format(ticker), tmp_df, allow_pickle=True)
        del df, tmp_df
        return

    def data_process_with_price(self, mp_mode=False):
        """
        :paramself.trans_record:
        :param silence_days: The number of days that Investment Trust companies have no action
        :param max_p: The maximum trading amount for "silence days"
        :param min_p: The minimum trading amount to be considered as Investment Trust's "Action"
        :param mode: 'exp' for use current data; 'update' for add new data; 'backtest' for re-processing the whole data
        :return:
        """
        if self.trans_record is None:
            self.trans_record = np.load(
                tw_path_setting(collapse='daily')[2] + 'tw_investment_trust_net_buy_sell_record_trans.npy',
                allow_pickle=True).item()
        price_path = tw_path_setting(collapse='daily')[0]
        self.bench_last_date = max(self.all_record)
        input_list = [
            [ticker, price_path, self.trans_record[ticker].copy()] for ticker in self.trans_record]
        if mp_mode:
            pool = mp.Pool(4)
            pool.map(self.it_bet_bs_job, input_list)
            pool.close()
        else:
            jj, total_len = 1, len(input_list)
            for input_i in input_list:
                jj = sub_process_bar(jj, total_len)
                self.it_bet_bs_job(input_i)
        return self.summary_dict

    def create_summary_dict(self):
        if not hasattr(self, 'end_year'):
            self.end_year = int(str(max(self.all_record))[:4])
        self.summary_dict = {yr: {} for yr in range(self.start_yr, self.end_year + 1)}
        jj, total_len = 1, len(self.trans_record)
        for ticker in self.trans_record:
            jj = sub_process_bar(jj, total_len)
            try:
                df = np.load(self.exp_save_dir + self.ticker_itbs_format.format(ticker), allow_pickle=True).item()
            except FileNotFoundError:
                continue
            df = pd.DataFrame(df).transpose().sort_index()
            df.index.name = 'date'
            df = df.reset_index(drop=False)
            if '收盤價' not in df:
                continue
            signal_df = df[df[self.signal_item] == 1]
            if len(signal_df) == 0:
                continue
            for yi in self.summary_dict:
                yr_signal = signal_df[(signal_df['date'] >= (yi * 10000)) & (signal_df['date'] <= (yi * 10000 + 9999))]
                if len(yr_signal) > 0:
                    self.summary_dict[yi][ticker] = {'return_h{}'.format(h): np.NAN for h in self.h_list}
                    for h in self.h_list:
                        self.summary_dict[yi][ticker]['return_h{}'.format(h)] = yr_signal['h{}_pct'.format(h)].tolist()
                    self.summary_dict[yi][ticker]['signal_date'] = yr_signal['date'].tolist()
        np.save(self.exp_save_dir + 'summary_{}_it_bet_bs_record.npy'.format(self.signal_item), self.summary_dict, allow_pickle=True)

    def summary_analysis(self, historical_thres=0.5, compare_bench=False):
        self.ticker_historical_perform_consider()
        self.summary_dict_add_condition()
        np.save(self.exp_save_dir + 'summary_{}_it_bet_bs_record.npy'.format(self.signal_item), self.summary_dict,
                allow_pickle=True)
        self.historical_thres = historical_thres
        self.compare_bench = self.compare_bench
        if (not hasattr(self, 'summary_dict')) or (self.summary_dict is None):
            self.summary_dict = np.load(self.exp_save_dir + 'summary_{}_it_bet_bs_record.npy'.format(self.signal_item),
                                        allow_pickle=True).item()
        self.bench = tw_price_df_loading(data_path=tw_path_setting(collapse='daily')[0], ticker='0050')
        self.bench = self.bench[self.bench['日期'] >= self.start]
        for h in self.h_list:
            self.bench['pct_h{}'.format(h)] = self.bench['收盤價'].pct_change(h).shift(-h)
            # 不買個股的話，錢放在0050，所以可以直接從收盤價做分母
        self.bench = self.bench.set_index('日期', drop=True)
        print('Historical win rate threshold={}'.format(self.historical_thres))
        for yr in self.summary_dict:
            hh = {'h{}'.format(hi): [] for hi in self.h_list}
            i = 0
            for ticker in self.summary_dict[yr]:
                if ('historical' in self.summary_dict[yr][ticker]) and (
                        self.summary_dict[yr][ticker]['historical'] < self.historical_thres):
                    continue
                tmp = self.ummary_dict[yr][ticker].copy()
                ni = len(tmp['signal_date'])
                flag = 0
                for h in self.h_list:
                    return_item = 'return_h{}'.format(h)
                    if (return_item in tmp) and (str(tmp[return_item][0]) != 'nan'):
                        hh['h{}'.format(h)] += tmp[return_item]
                        flag = 1
                i += flag
            print('year={}, n={}'.format(yr, i))
            for hj in [1, 3, 5, 10]:
                h_tmp = hh['h{}'.format(hj)]
                print('h{} return={}'.format(hj, round(np.nanmean(h_tmp), 3)),
                      '|win0={}'.format(round(len([m for m in h_tmp if m > 0]) / len(h_tmp), 3)),
                      'win5={}'.format(round(len([m for m in h_tmp if m > 0.05]) / len(h_tmp), 3)),
                      'win10={}'.format(round(len([m for m in h_tmp if m > 0.1]) / len(h_tmp), 3)))
            print('-----------------------------')
        if self.compare_bench:
            for yr in self.summary_dict:
                hh = {'h{}'.format(hi): [] for hi in [1, 2, 3, 5, 10]}
                i = 0
                for ticker in self.summary_dict[yr]:
                    if ('historical' in self.summary_dict[yr][ticker]) and (
                            self.summary_dict[yr][ticker]['historical'] < self.historical_thres):
                        continue
                    tmp = self.summary_dict[yr][ticker].copy()
                    for sig_date in tmp['signal_date']:
                        bench_pct = {hi: self.bench.loc[sig_date, 'pct_h{}'.format(hi)] for hi in self.h_list}
                        for hi in self.h_list:
                            if str(bench_pct[hi]) != 'nan':
                                tmp['return_h{}'.format(hi)] = [tmp['return_h{}'.format(hi)][0] - bench_pct[hi]]

                    flag = 0
                    for h in self.h_list:
                        return_item = 'return_h{}'.format(h)
                        if (return_item in tmp) and (str(tmp[return_item][0]) != 'nan'):
                            hh['h{}'.format(h)] += tmp[return_item]
                            flag = 1
                    i += flag
                print('year={}, n={}'.format(yr, i))
                for hj in [1, 3, 5, 10]:
                    h_tmp = hh['h{}'.format(hj)]
                    print('h{} return-bench={}'.format(hj, round(np.nanmean(h_tmp), 3)),
                          '|win0={}'.format(round(len([m for m in h_tmp if m > 0]) / len(h_tmp), 3)))
                print('-----------------------------')

    def load_summary_dict(self):
        if (not hasattr(self, 'summary_dict')) or (self.summary_dict is None):
            self.summary_dict = np.load(self.exp_save_dir + 'summary_{}_it_bet_bs_record.npy'.format(self.signal_item),
                                        allow_pickle=True).item()
        return self.summary_dict

    def ticker_historical_perform_consider(self):
        self.summary_dict = self.load_summary_dict()
        for yr in self.summary_dict:
            if yr == min(self.summary_dict):
                continue
            for ticker in self.summary_dict[yr]:
                history = []
                for yh in range(self.start_yr, yr):
                    if ticker in self.summary_dict[yh]:
                        history.append(np.nanmean(self.summary_dict[yh][ticker]['return_h3']))
                if len(history) > 0:
                    self.summary_dict[yr][ticker]['historical'] = len([r for r in history if r > 0]) / len(history)

    def current_signal(self):
        self.summary_dict = self.load_summary_dict()
        today_year = datetime.datetime.today().year
        if today_year not in self.summary_dict:
            print('No Signal in Year {}'.format(today_year))
        else:
            defa = {ticker: max(self.summary_dict[today_year][ticker]['signal_date']) for ticker in self.summary_dict[today_year]}
            maxt = [t for t in defa if defa[t] >= max([defa[t] for t in defa])]
            for t in maxt:
                print('\n', t, defa[t])
        return

    def check_ticker_it_record(self, ticker='8255'):
        if os.path.isfile(self.exp_save_dir + self.ticker_itbs_format.format(ticker)):
            df = np.load(self.exp_save_dir + self.ticker_itbs_format.format(ticker), allow_pickle=True).item()
            df = pd.DataFrame(df)
            df = df.set_index('date', drop=True)
        else:
            print('Cannot find', self.exp_save_dir + self.ticker_itbs_format.format(ticker))
            df = None
        return df

    def summary_dict_add_condition(self):
        jj, total_step = 1, len(self.summary_dict)
        for yr in self.summary_dict:
            jj = sub_process_bar(jj, total_step)
            for ticker in self.summary_dict[yr]:
                try:
                    df = np.load(self.exp_save_dir + self.ticker_itbs_format.format(ticker), allow_pickle=True).item()
                except FileNotFoundError:
                    continue
                df = pd.DataFrame(df).transpose().sort_index()
                df.index.name = 'date'
                tmp = self.summary_dict[yr][ticker].copy()
                signal_date_list = tmp['signal_date']
                signal_date_intra_pct = []
                net_volume = []
                mktcap = []
                h1open = []
                df['next_date_open'] = df['開盤價'].shift(-1)

                for signal_date in signal_date_list:
                    signal_date_intra_pct.append(df.loc[signal_date, '收盤價'] / df.loc[signal_date, '開盤價'] - 1)
                    net_volume.append((df.loc[signal_date, '買賣超股數'] / 1000) / df.loc[signal_date, '成交量'])
                    mktcap.append(df.loc[signal_date, '總市值(億)'])
                    h1open.append(df.loc[signal_date, 'next_date_open'] / df.loc[signal_date, '收盤價'])
                tmp['nextopen_todayclose'] = h1open
                tmp['signal_date_intra_pct'] = signal_date_intra_pct
                tmp['net_volume'] = net_volume
                tmp['mktcap'] = mktcap
                self.summary_dict[yr][ticker] = tmp

    def main(self):
        self.all_record = self.data_loading()
        self.trans_record = self.itntbs_record_transform(save=True)
        self.supplement_from_goodinfo()
        # 需下載日收盤還原報表
        # summary_dict, signal_item = data_process_with_price(trans_record=trans_record, silence_days=25, max_p=2000, min_p=100000, mode='backtest')
        self.data_process_with_price()
        self.summary_dict = self.create_summary_dict()
        self.current_signal()
        self.summary_analysis()
        self.current_signal()

    def quick_see_if_signal_today(self, today=None):
        """
        :param today: int, YYYYmmdd
        :return:
        """
        if today is None:
            today = max(self.all_record)
        print('\nToday = ', today)
        date_list = sorted(list(self.all_record.keys()))
        if today not in date_list:
            print('{} is not a valid stock date. Please try another date.'.format(today))
            return
        today_idx = date_list.index(today)
        silence_start = date_list[today_idx - self.silence_days]
        for ticker in self.trans_record:
            if today in self.trans_record[ticker]:
                if self.trans_record[ticker][today]['買賣超股數'] >= self.min_p:
                    during_date = [d for d in self.trans_record[ticker] if (d >= silence_start) and (d < today)]
                    sum_p = 0
                    for d in during_date:
                        # if self.trans_record[ticker][d]['買賣超股數'] > 0:
                            # sum_p += self.trans_record[ticker][d]['買賣超股數']
                        sum_p += abs(self.trans_record[ticker][d]['買賣超股數'])
                    if sum_p <= self.max_p:
                        mktcap = get_mktcap_yahoo(stock=ticker)
                        # try:
                        #     price_data = tw_price_df_loading(data_path=tw_path_setting(collapse='daily')[0], ticker=ticker,
                        #                                      collapse='daily')
                        #     mktcap = price_data.iloc[-1]['總市值(億)']
                        # except:
                        #     mktcap = np.NAN
                        print(today, ticker, '總市值(億)={}'.format(mktcap))
        print('\n====================================================')

    def daily_main(self, today=None):
        self.mode = 'backtest'
        self.data_loading()
        self.itntbs_record_transform(save=True)
        # self.data_process_with_price()
        # self.current_signal()
        self.quick_see_if_signal_today(today=today)

#################################################################
# Evaluate strategy
    def calculate_return(self):
        # use data in self.trans_record, execute daily_main
        # how to get the price (unadjusted)
        signal_record = {}
        print('Stock LB of ITBS buy = {}'.format(self.min_p))
        for ticker in self.trans_record:
            t_date_list = sorted(list(self.trans_record[ticker]),reverse=False)
            for t_date in self.trans_record[ticker]:
                if t_date_list.index(t_date) < self.silence_days:
                    continue
                silence_start = t_date_list[t_date_list.index(t_date) - self.silence_days]
                if self.trans_record[ticker][t_date]['買賣超股數'] >= self.min_p:
                    sum_p = sum([abs(self.trans_record[ticker][d]['買賣超股數']) for d in self.trans_record[ticker] if (d >= silence_start) and (d < t_date)])
                    if sum_p <= self.max_p:
                        '''
                        1. get mktcap from goodinfo or other hostorical record (?
                        2. see if mktcap < 100
                        '''
                        if ticker not in signal_record:
                            signal_record[ticker] = {'signal_dates':[t_date]}
                        else:
                            signal_record[ticker]['signal_dates'].append(t_date)
        return signal_record

    def rebuild_signal_record(self):
        signal_record = self.calculate_return()
        for ticker in signal_record:
            try:
                signal_record[ticker]['mktcap'] = get_mktcap_yahoo(ticker)
            except:
                print('{} cannot get marketcap value from Yahoo'.format(ticker))
                signal_record[ticker]['mktcap'] = np.NAN
        np.save('tw_data/itbs_signal_record.npy', signal_record, allow_pickle=True)

    def collect_return_rate(self, signal_record, stock, hold_days):
        for signal_date in signal_record[stock]['signal_dates']:
            if signal_date not in signal_record[stock]['trading_h{}'.format(hold_days)]:
                time.sleep(random.randint(5,10))
                rr = calculate_return_yahoo(stock, signal_date, hold_days=hold_days)
                signal_record[stock]['trading_h{}'.format(hold_days)][signal_date] = rr
        return signal_record

    def show_avg_return_simple(self, signal_record, hold_days, mktcap_limit=100):
        avg_rr, n = 0, 0
        for stock in signal_record:
            if signal_record[stock]['mktcap'] <= mktcap_limit:
                tmp = [signal_record[stock]['trading_h{}'.format(hold_days)][sd] for sd in
                       signal_record[stock]['signal_dates']]
                tmp = [t for t in tmp if str(t) != 'nan']
                avg_rr += np.nansum(tmp)
                n += len(tmp)
        avg_rr = avg_rr / n
        print('average return rate = {}; mktcap limit = {}'.format(avg_rr, mktcap_limit))

    def easy_show_return(self, hold_days=0):
        signal_record = np.load('tw_data/itbs_signal_record.npy', allow_pickle=True).item()
        jj, total_len = 0, len(signal_record)
        for stock in signal_record:
            jj = sub_process_bar(jj, total_len)
            signal_record[stock]['trading_h{}'.format(hold_days)] = signal_record[stock].get('trading_h{}'.format(hold_days), {})
            signal_record = self. collect_return_rate(signal_record, stock, hold_days)
        np.save('tw_data/itbs_signal_record.npy', signal_record, allow_pickle=True)
        self.show_avg_return_simple(signal_record, hold_days)


############################################
def get_mktcap_yahoo(stock):
    url = 'https://tw.stock.yahoo.com/quote/{}.TW/profile'.format(stock)
    resp = requests.get(url, headers={
        'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like gecko) Chrome/63.0.3239.132 Safari/537.36'})
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    mkt_num = ''
    mkt_num_index = soup.text.index('市值 (百萬)') + 7
    while soup.text[mkt_num_index] != '市':
        mkt_num += soup.text[mkt_num_index]
        mkt_num_index += 1
    if '-' in mkt_num:
        # ETF
        return -1
    else:
        return float(mkt_num.replace(',', '')) / 100


def get_yahoo_price(stock, signal_date):
    url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date={}&stockNo={}'.format(signal_date, stock)
    try:
        price_data = pd.read_html(requests.get(url).text)[0]
    except:
        print('\nError with {} on {}'.format(stock, signal_date))
        return None
    price_data.columns = price_data.columns.droplevel()
    price_data.columns = ['date', 'volume', 'transfer_value', 'open', 'high', 'low', 'close', 'close_change','volume-time']
    return price_data


def calculate_return_yahoo(stock, signal_date, hold_days=0):
    signal_datetime = datetime.datetime.strptime(str(signal_date), '%Y%m%d')
    buy_date = signal_datetime + BDay(1)
    sell_date = signal_datetime + BDay(hold_days + 1)
    if buy_date.month == sell_date.month:
        price_df = get_yahoo_price(stock, buy_date.strftime('%Y%m%d'))
    else:
        price_df = pd.concat([get_yahoo_price(stock, buy_date.strftime('%Y%m%d')), get_yahoo_price(stock, sell_date.strftime('%Y%m%d'))])
    if price_df is None:
        return np.NAN
    price_df = price_df[price_df['volume'] > 0]
    price_df['date'] = price_df['date'].apply(lambda x: int(19110000 + int(x.replace('/', ''))))
    price_df = price_df.set_index('date', drop=True)
    if int(buy_date.strftime('%Y%m%d')) not in price_df.index:
        if int(buy_date.strftime('%Y%m%d')) > max(price_df.index):
            price_df = get_yahoo_price(stock, (signal_datetime + BDay(5)).strftime('%Y%m%d'))
            if price_df is None:
                return np.NAN
            price_df = price_df[price_df['volume'] > 0]
            price_df['date'] = price_df['date'].apply(lambda x: int(19110000 + int(x.replace('/', ''))))
            price_df = price_df.set_index('date', drop=True)
        buy_d = min([d for d in price_df.index if d > int(buy_date.strftime('%Y%m%d'))])
    else:
        buy_d = int(buy_date.strftime('%Y%m%d'))
    buy_price = price_df.loc[buy_d, 'open']
    if int(sell_date.strftime('%Y%m%d')) not in price_df.index:
        sell_d = min([d for d in price_df.index if d > int(sell_date.strftime('%Y%m%d'))])
    else:
        sell_d = int(sell_date.strftime('%Y%m%d'))
    sell_price = price_df.loc[sell_d, 'close']
    pct = float(sell_price)/float(buy_price) - 1
    return pct


def save_price_data(price_data, stock, save_path='tw_data/price/'):
    file_name = '{}_{}-{}'.format(stock, price_data.index[0], price_data.index[-1])
    price_data = {c: price_data[c] for c in price_data.columns}
    np.save(save_path + file_name, allow_pickle=True)

# 1. 用每次需要時再下載的方法
# 2. 同1, 但先去既有的data裡面找，如果沒有，再下載合併上

########################################
def regression_exp(summary_dict, x_item, y_item):
    x, y = [], []
    for yr in summary_dict:
        for ticker in summary_dict[yr]:
            if summary_dict[yr][ticker]['mktcap'][0] > 100:
                continue
            # if summary_dict[yr][ticker]['nextopen_todayclose'][0] > 1.03:
            #     continue
            for i in range(len(summary_dict[yr][ticker]['signal_date'])):
                x.append(summary_dict[yr][ticker][x_item][i])
                y.append(summary_dict[yr][ticker][y_item][i])


def plotly_condition_exp(signal_item, summary_dict, x_item='signal_date_intra_pct', y_item='return_h3'):
    import plotly.graph_objs as go
    from plotly.offline import plot
    x, y, text_ = [], [], []
    for yr in summary_dict:
        for ticker in summary_dict[yr]:
            if summary_dict[yr][ticker]['mktcap'][0] > 100:
                continue
            # if summary_dict[yr][ticker]['nextopen_todayclose'][0] > 1.03:
            #     continue
            for i in range(len(summary_dict[yr][ticker]['signal_date'])):
                x.append(summary_dict[yr][ticker][x_item][i])
                y.append(summary_dict[yr][ticker][y_item][i])
                text_.append('{}, {}'.format(ticker, summary_dict[yr][ticker]['signal_date'][i]))
    data = go.Scatter(
        x=x,
        y=y,
        name='{}-{}'.format(x_item, y_item),
        mode='markers',
        text=text_
    )
    shape_list = []
    layout = go.Layout(
        title=' TE-ITBS:{} | {}-{}'.format(signal_item, x_item, y_item),
        xaxis=dict(
            title=x_item
        ),
        yaxis=dict(
            title=y_item
        ),
        shapes=shape_list
    )
    fig = go.Figure(data=data, layout=layout)
    auto_open = itbs_exp_dir_setting().startswith('tw_data/')
    try:
        plot_url = plot(fig, filename=itbs_exp_dir_setting() + 'TW-ITBS_exp_{}_{}-{}.html'.format(signal_item, x_item,
                                                                                                  y_item), auto_open=auto_open)
    except:
        from plotly.offline import plot
        plot_url = plot(fig, filename=itbs_exp_dir_setting() + 'TW-ITBS_exp_{}_{}-{}.html'.format(signal_item, x_item,
                                                                                                  y_item), auto_open=auto_open)


########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == '__main__':
    itbs = ITBS()
    itbs.daily_main()
