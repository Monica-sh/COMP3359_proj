import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta


def load_env(ticker='aapl'):
    data_path = "Dataset/Stocks/{}.us.txt".format(ticker) 
    df = pd.read_csv(data_path)
    df.index = df["Date"]
    df = df.drop(['Date'], axis=1)
    return Environment(df)

class Environment:
    def __init__(self, raw_df):
        '''
        data_df: 6 columns [Open, High, Low, Close, Volume, OpenInt] and Index with Date.

        state: 
        
        one postition storing number of shares, buy one time = 1 share added
        Reminder: all date is represented by the string
        '''
        self.raw_df = raw_df
        self.start = raw_df.index[0]
        self.end = raw_df.index[-1]
        self.state_shape = 6

        # self.data_df = None
        # self.state = np.zeros(self.state_shape)
        # self.holding_stocks = False
        # self.buy_price = None
        # self.buy_date = None

        self.date = self.start
        self.reset()
    
    def reset(self, date=None):
        if date == None:
            self.date = self.start
        else:
            while date not in self.raw_df.index:
                date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            self.date = date
        
        self.process_data()
        self.state = self.data_df.loc[self.date]
        self.holding_stocks = False
        self.buy_price = None
        self.buy_date = None

        return self.state
    
    # construct data set after defining self.date
    def process_data(self):
        #normalise data
        data_df = self.raw_df[self.date:]
        df_max = np.max(data_df, axis=0)
        self.data_df = (data_df / df_max).fillna(0)

    def step(self, action):
        '''
        input: action as an int list, positive = buy, negative = sell
        '''
        if self.date == self.end:
            # close all positions and return done == True 
            reward = self.calculate_pnl(action)
            return None, reward, True
        
        reward = self.calculate_pnl(action)
        idx = np.searchsorted(self.data_df.index, self.date)
        self.date = self.data_df.index[idx+1]
        self.state = self.data_df.loc[self.date]

        return self.state, reward, False

    def calculate_pnl(self, action):
        date_index = self.date
        open = self.state["Open"]
        close = self.state["Close"]
        # print(close, action, self.state)
        
        # take the but/sell action
        # action = 2 >> buy, action = 1 >> no sell no buy, action = 0 >> sell
        reward = 0 #TODO this line seems can be deleted 

        if action == 0 and self.holding_stocks == True:  # The agent has sold the stocks
            self.holding_stocks = False
            reward = close - self.buy_price
        elif action == 2 and self.holding_stocks == False:  # Buy some stocks
            self.holding_stocks = True
            self.buy_price = open
            self.buy_date = self.date
            reward = None
        else:  # No-op
            # If the agent is holding stocks, set the reward as None and
            # the agent will handle reward calculation later. If the agent
            # is not holding stocks and doesn't intend to buy, the reward is 0.
            reward = None if self.holding_stocks else 0

        return reward