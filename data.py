import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_env(ticker='aapl'):
    data_path = "Dataset/Stocks/{}.us.txt".format(ticker) 
    df = pd.read_csv(data_path)
    df.index = df["Date"]
    df = df[["Open","Close"]]
    return Environment(df)

class Environment:
    def __init__(self, data_df):
        '''
        state: one postition storing number of shares, buy one time = 1 share added
        Reminder: all date is represented by the string
        '''
        self.data_df = data_df
        self.start = data_df.index[0]
        self.end = data_df.index[-1]
        self.state_shape = 1
        self.state = np.zeros(self.state_shape)
        self.date = self.start
    
    def reset(self, date=None):
        if date == None:
            self.date = self.start
        else:
            while not self.data_df.loc[data]:
                date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            self.date = date
        self.state = np.zeros(self.state_shape)

        return self.state

    def step(self, action):
        '''
        input: action as an int list, positive = buy, negative = sell
        '''
        if self.date == self.end:
            # close all positions and return done == True 
            reward = self.calculate_pnl(action)
            return np.ndarray(-1), reward, True
        
        reward = self.calculate_pnl(action)
        idx = np.searchsorted(self.data_df.index, self.date)
        self.date = self.data_df.index[idx+1]

        return self.state, reward, False

    def calculate_pnl(self, action):
        date_index = self.date
        # open = self.data_df.loc[date_index]["Open"]
        close = self.data_df.loc[date_index]["Close"]
        print(close, action, self.state)
        
        # take the but/sell action
        # action = 2 >> buy, action = 1 >> no sell no buy, action = 0 >> sell
        if self.state[0] == 0 and action == -1:
            # cannot short sell
            return 0
        
        self.state += action
        return - (action - 1) * close


# def load_env(tickers=['aapl']):
#     data_df = None
#     for i in range(len(tickers)):
#         data_path = "Dataset/Stocks/{}.us.txt".format(tickers[i]) 
#         df = pd.read_csv(data_path)
#         df = df[["Date","Open","Close"]]
#         # print(df.head(), df.iloc[2,0], type(df.iloc[2,0]))
#         if data_df == None:
#             df = df.rename(columns={"Open": "Open1", "Close": "Close1"}, errors="raise")
#             data_df = df
#         else:
#             data_df = pd.merge(data_df, df, how='inner', on=['Date'], suffixes=['', str(i)])

#     return Environment(data_df, len(tickers))


# class Environment:
#     def __init__(self, data_df, num_stocks):
#         '''
#         state: 2d of current position & initial capital inputs
#                 e.g. [[0,10,-20],
#                       [0, -150, 400]]   
#                 means holding 0 shares in stock[0],
#                 10 shares in stock[1] & spent 150 dollars,
#                 short-sold 20 shares in stock[2] & sold for 400 dollars.
#         '''
#         self.data_df = data_df
#         self.start = datetime.strptime(data_df.iloc[0,0], '%Y-%m-%d')
#         self.end = datetime.strptime(data_df.iloc[-1,0], '%Y-%m-%d')
#         self.state_shape = (2, num_stocks)

#         self.state = np.zeros(self.state_shape)
#         self.date = self.start
    
#     def reset(self, date=None):
#         if date == None:
#             self.date = self.start
#         else:
#             while not self.data_df.loc[data]:
#                 date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
#             self.date = datetime.strptime(date, '%Y-%m-%d')
#         self.state = np.zeros(self.state_shape)

#     def process_data(self):
#         pass

#     def step(self, actions):
#         '''
#         input: action as an int list, positive = buy, negative = sell
#         '''
#         if self.date == self.last:
#             # close all positions and return done == True
#             actions = - self.state 
#             reward = self.calculate_pnl(actions)
#             return None, reward, done
        
#         reward = self.calculated_pnl(actions)

#         self.date += timedelta(days=1)
        
#         return self.state, reward, done

#     def calculate_pnl(self, actions):
#         date_index = self.date.strftime('%Y-%m-%d')
#         prices = self.data_df.loc[date_index]
#         open = []
#         close = []
#         for i in range(self.state_shape[1]):
#             open.append(prices[i * 2])
#             close.append(prices[i * 2 +1])
        
#         # take the but/sell action
#         i = 0
#         reward = 0
#         for a in actions:
#             if a == 0:
#                 continue
#             elif a > 0:
#                 if self.state[0,i] >= 0:
#                     #simply adding long position

#                     self.state[1,i] -= open[i] * a
#                 elif self.state[0,i] < -a:
#                     # shortselling not closed
#                     rewards += (a / (- self.state[0,i])) * self.state[1,i] - open[i] * a
#                     self.state[1,i] = (1 - a / (- self.state[0,i])) * self.state[1,i]

#                 else:
#                     # shortselling postition closed and will long postition left
#                     reward += self.state[1,i] - open[i] * (- self.state[0,i])
#                     self.state[1,i] = - open[i] * (a - self.state[0,i])

#             else:
#                 if self.state[0,i] < 0:
#                     # simply add short position

#                     self.state[1,i] += close[i] * a
#                 elif self.state[0,i] > -a :
#                     # long position is not closed
#                     reward += close[i] * a - self.state[1,i] * ( - a /self.state[0,i])
#                     self.state[1,i] = self.state[1,i] * (1 + a /self.state[0,i])
#                 else:
#                     # long postion closed
#                     reward += close[i] * (self.state[0,i]/ (- a)) - self.state[1,i]
#                     self.state[1,i] = close[i] * (a + self.state[0,i])
                
#             i+=1
        
#         self.state[0] += actions
#         return reward
