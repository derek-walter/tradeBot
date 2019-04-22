import sys 
import random
import pandas as pd
import numpy as np
from collections import deque

# Functions
def relu(x):
    if x > 0:
        return x
    else:
        return 0.01*x

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def xabx(x):
    return x/(1 + np.abs(x))

def testing123(model, NN_input_size, reset=False):
    '''Given a model and input dimensions, this generates data and retuns a history object.
    Parameters: model, the same NN_input_size used in Bots.py, reset model weights
    Returns: History object
    '''
    # Dummy Data
    X_train = np.random.random((1000, *NN_input_size))
    y_train = np.random.random((1000, NN_input_size[-1]))
    historyObject = model.fit(X_train, y_train, batch_size=100, epochs=1, validation_split=0.25)
    if reset:
        model.reset_states()
    return historyObject

def Train(bot, scaler, train_data, episode_count=3, shares=0, start_cash=20000, replay_size=32):
    '''Notes of interest: This training procedure takes a cursor to train data, uses HER to
    Learn from past rewards. To summarize DQN, we just predict as we step through data.
    Periodically (I chose every replay_size steps), we fit on a batchsize of memories.
    Parameters:
        bot: actual capable bot object
        scaler: some scaler object fit to data coming in
        train_data: Cursor to specific db data (like output query of train data).
        window: Legth of timesteps for bot
        shares: Number of starting stock shares for little bot
        cash: amount of starting cash
        replay_size: Batch size for HER
    '''
    window = bot.NN_input_shape[0]
    if window > replay_size:
        raise ValueError('Window cannot be larger than replay size.')
    state_vars = ['change', 'close_vwap', 'high_low', 'open_close', 'volume']
    # Create Log
    this_log = pd.DataFrame(columns = ['Loss', 'Reward', 'Epsilon'])
    # Routine Setup
    epsilon = 0.99
    discount = 0.95
    options = ['buy', 'sell', 'hold']
    for episode in range(episode_count):
        state = np.zeros((1, window, len(state_vars) + 2)) # Allocate memory. This is of (1, #, #) for single fitting
        share_prices = deque([]) # Time Com. of O(1) for left popping...
        #train_data.rewind()
        cash = start_cash
        count = 0
        done = False
        for item in train_data:
            curr_price = item['data']['close']
            curr_change = item['data']['change']
            # End Conditions (can't buy & can't sell)
            if cash < curr_price and len(share_prices) == 0:
                done = True
            # Simple window: Have state going back in time, save that, shift down, ammend this step
            previous_state = state
            state[0][1:,:] = state[0][:-1, :]
            # Generating State Vars
            if cash > curr_price:
                can_buy = 1
            else:
                can_buy = 0
            if len(share_prices) >= 1:
                can_sell = 1
            else:
                can_sell = 0
            # NOTE: Scaler Transform requires alphabetical columns for proper prediction
            temp_state = [scaler.transform([[item['data'][j] for j in state_vars]])[0]]
            temp_state = np.array([np.append(temp_state[0], can_buy)])
            temp_state = np.array([np.append(temp_state[0], can_sell)])
            state[0][0,:] = temp_state
            print('State: \n', state)
            if count >= window: # Start Bot once state is full enough
                actions = bot.predict(previous_state)
                # Exploration
                if random.random() < epsilon:
                    action = random.randrange(len(options))
                    choice = options[action]
                else:
                    # Greedy chooser, best action in bots world
                    action = np.argmax(actions)
                    choice = options[action]
                # Exploration Decay
                if epsilon > 0.1:
                    epsilon -= 0.01
                    print('Epsilon: ', round(epsilon, 5))
                else:
                    epsilon = 0.1
                # Reward Engineering [Buy, Sell, Hold] (0, 1, 2)
                if choice=='buy' and cash > curr_price: # Buy
                    cash -= curr_price
                    shares += 1
                    share_prices.append(curr_price)
                    reward = 0
                elif choice=='sell' and shares > 0: # Sell
                    cash += curr_price
                    shares -= 1
                    profit = curr_price - share_prices.popleft()
                    reward = relu(profit) # Future Action-Value needs to be appended in HER
                else: # Hold
                    reward = curr_change*0.1
                # It is important to note the memory deque is 1000 long.
                bot.memory.append((previous_state, action, reward, state, done))
                # Hindsight Experience Replay
                if (count - window + 1) % replay_size == 0:
                    print('Replaying...')
                    batch = random.sample(bot.memory, replay_size)
                    for batch_state, batch_action, batch_reward, batch_new_state, batch_done in batch:
                        # Current buy, sell, hold predictions, Q(s, a)
                        targets = bot.predict(batch_state)
                        # Expected buy, sell, hold predictions
                        action_value = bot.predict(batch_new_state)
                        if not batch_done:
                            # Adjusting target by return and discounted future return
                            # Must += instead of reassigning rewards because reward could be zero or negative
                            targets[0][batch_action] += batch_reward + discount*np.max(action_value)
                        else:
                            # Just saving the return
                            targets[0][batch_action] += batch_reward
                        history = bot.fit(batch_state, targets, batch_size=1, epochs=1)
                        this_log = this_log.append({'Loss':history.history['loss'][0], 'Reward':batch_reward, 'Epsilon':round(epsilon, 5)}, ignore_index = True)
            count += 1
        else:
            '''Wow! It Made it!'''
            print('The Bot Survived.')
        print('Episode: ', episode)
    return bot, this_log

def Test(bot, scaler, test_data, shares=0, start_cash=20000):
    '''Notes of interest: This training procedure takes a cursor to train data, uses HER to
    Learn from past rewards. To summarize DQN, we just predict as we step through data.
    Periodically (I chose every replay_size steps), we fit on a batchsize of memories.
    Parameters:
        bot: actual capable bot object
        scaler: some scaler object fit to data coming in
        train_data: Cursor to specific db data (like output query of train data).
        window: Legth of timesteps for bot
        shares: Number of starting stock shares for little bot
        cash: amount of starting cash
        replay_size: Batch size for HER
    '''
    window = bot.NN_input_shape[0]
    state_vars = ['change', 'close_vwap', 'high_low', 'open_close', 'volume']
    # Create Log
    portfolio_log = pd.DataFrame(columns = ['Action', 'Reward', 'Shares', 'Cash', 'Close'])
    # Routine Setup
    options = ['buy', 'sell', 'hold']
    state = np.zeros((1, window, len(state_vars) + 2))
    share_prices = deque([]) # Time Com. of O(1) for left popping...
    #test_data.rewind()
    cash = start_cash
    profits = 0
    count = 0
    done = False
    for item in test_data:
        curr_price = item['data']['close']
        curr_change = item['data']['change']
        # Simple window: Have state going back in time, save that, shift down, ammend this step
        previous_state = state
        state[0][1:,:] = state[0][:-1, :]
        # Generating State Vars
        if cash > curr_price:
            can_buy = 1
        else:
            can_buy = 0
        if len(share_prices) >= 1:
            can_sell = 1
        else:
            can_sell = 0
        # NOTE: Scaler Transform requires alphabetical columns for proper prediction
        temp_state = [scaler.transform([[item['data'][j] for j in state_vars]])[0]]
        temp_state = np.array([np.append(temp_state[0], can_buy)])
        temp_state = np.array([np.append(temp_state[0], can_sell)])
        state[0][0,:] = temp_state
        if count >= window and not done: # Start Bot once state is full enough
            actions = bot.predict(previous_state)
            action = np.argmax(actions)
            choice = options[action]
            # No Exploration
            # Reward Engineering [Buy, Sell, Hold] (0, 1, 2)
            if choice=='buy' and cash > curr_price: # Buy
                cash -= curr_price
                shares += 1
                share_prices.append(curr_price)
            elif choice=='sell' and shares > 0: # Sell
                cash += curr_price
                shares -= 1
                profits += curr_price - share_prices.popleft()
            else:
                profits += curr_change
        count += 1
        if cash < curr_price and len(share_prices) == 0:
            done = True
        portfolio_log = portfolio_log.append({'Action':choice, 'Shares':shares, 'Cash':cash, 'Profits':profits, 'Close':curr_price}, ignore_index = True)
    else:
        '''Wow! It Made it!'''
        print('The Bot Survived.')
    return portfolio_log


if __name__ == "__main__":
    import pandas as pd
    from sklearn.externals import joblib
    sys.path.append("./src")
    from Bots import Bot_LSTM
    from mongo import sstt_cursors
    # Specifics (Note alphabetic...for scaler)
    state_vars = ['change', 'close_vwap', 'high_low', 'open_close', 'volume']
    bot = Bot_LSTM((14, len(state_vars)+2))
    scaler = joblib.load("resources/tech_scaler.pkl")
    '''Train'''
    #for stock in stocks:
    train_cursor, test_cursor = sstt_cursors('AAPL')
    # The flow of train is to train a bot on a stock and get back the bot, with a PD.DataFrame log
    # Ideally this would continue for numerous stocks for one bot.
    episodes = 3
    bot, train_log = Train(bot, scaler, train_cursor)
    '''Test'''
    portfolio_log = Test(bot, scaler, test_cursor)
    '''
    x Make scaler on all stocks
    x Download Stock Data to Dict
    x Insert into MongoDB

    For each stock:
        generate train, test
        train
            add log to main log
        test
            add to test log
    '''
