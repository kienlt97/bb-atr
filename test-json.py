
import numpy as np 

if __name__ == '__main__':
    cummulative_buy = [1]
    cummulative_sell = []

    profit = np.sum(cummulative_sell) - np.sum(cummulative_buy)
    print(profit)