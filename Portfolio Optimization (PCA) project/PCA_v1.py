#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from numpy.core._multiarray_umath import ndarray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import scipy
from datetime import timedelta, datetime
#%%



#%%
def load_data(start_time) -> object:
    start = datetime.strftime(start_time, '%Y%m%d')
    # address1 = '/Users/apple/Desktop/selectedstock/' + start + 'top.txt'
    address1 = 'C:\\Users\\Tianyi Zhang\\Desktop\\Quant Method 5220\\selectedstock\\' + start + 'top.txt'
    namelist1 = pd.read_csv(address1, delim_whitespace = True)
    # address2 = '/Users/apple/Desktop/selectedstock/' + start + 'bot.txt'
    address2 = 'C:\\Users\\Tianyi Zhang\\Desktop\\Quant Method 5220\\selectedstock\\' + start + 'bot.txt'
    namelist2 = pd.read_csv(address2, delim_whitespace = True)
    name = namelist1['Permnoline.postop.'].append(namelist2['Permnoline.posbot.'])


    # 1. add 10 stocks for allocation testing
    dic = dict()

    for x in name:
        address = '/Users/apple/Desktop/PycharmProjects/Quant_Method/' + x + '.txt'
        address = 'C:\\Users\\Tianyi Zhang\\Desktop\\Quant Method 5220\\stock_price\\' + x + '.txt'
        dic[int(x)] = pd.read_csv(address, delimiter=',')
        dic[int(x)].index = pd.to_datetime(dic[int(x)].date.astype(str))
        dic[int(x)] = dic[int(x)][['PERMNO', 'TICKER', 'PRC']]

    # 2. get rid of '-' sign by abs()
    for x in dic.keys():
        dic[x].PRC = abs(dic[x].PRC)

    # 3. calculate return rate and log_return
    end_time = start_time + one_week
    end_time = datetime.strftime(end_time, '%Y%m%d')
    test_date_index = dic[10057].index[dic[10057].index < end_time]
    return_df = pd.DataFrame(index=test_date_index)
    for x in dic.keys():
        dic[x]['return'] = dic[x].PRC.pct_change() + 1
        dic[x]['return'].fillna(0, inplace=True)
        return_df[x] = dic[x]['return']
    return_df = return_df[1:]

    log_return_df = return_df.apply(np.log)

    return_df.shape



    return log_return_df



#%%
def explain_model_fitting(log_return_df):
    # 1. PCA implementation--choose the number of component
    pca_test = PCA(n_components=10)
    pca_test.fit(log_return_df)

    # find the reasonable value of component
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_) * 100)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance ratio')
    plt.show()

    # then set up component to be 5
    pca = PCA(n_components=5)
    pca.fit(log_return_df)
    portfolio1 = pca.fit_transform(log_return_df)

    # 2. compare different methods. Ideally in this step, we can get without out-sample would be a better estimation
    # use in sample and out of sample data
    data_in, data_out = train_test_split(log_return_df, test_size=0.3, random_state=0)
    pca2 = PCA(n_components=5)
    pca2.fit(data_in)
    portfolio2 = pca2.fit_transform(data_in)

    p1_r = []
    p2_r = []
    for i in range(5):
        r1 = np.dot(log_return_df[i - 5:], portfolio1)
        p1_r.append(r1)
        r2 = np.dot(log_return_df[i - 5:], portfolio2)
        p2_r.append(r2)

    plt.plot(p1_r, label='without out_of_sample', color="red")
    plt.plot(p2_r, label='out_of_sample', color="black")
    plt.show()
    # then the return without splitting sample should be larger and we determine not to split

    # #### since we set up n_component = 5, we get PCs after implementation.
    # #### vec is the 5 x 10 eigenvector matrix
    vec = pca.components_
    print(vec)

    # #### val is the eigenvalue vector
    val = np.diag(pca.explained_variance_)
    print(val)

    # #### cov is the data covariance matrix 10 by 10
    cov = pca.get_covariance()
    print(cov)

    # #### make sure that (cov * eigenvector = eigenvalue * eigenvector ) holds
    test1 = vec @ cov @ vec.T
    test2 = np.diag(val)
    np.allclose(test1, val)
    print(test1 == test2)

    # this step is to give us a basic impression of how different components influence our portfolio
    # I suppose this step is useful  to illustrate that we have assets in different industry to diversify portfolio
    portfolio = pca.inverse_transform()
    for i in range(3):
        plt.scatter(portfolio[:, 2 * i], portfolio[:, 2 * i + 1], c=portfolio.target, edgecolor='none',
                    alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
        plt.xlabel('component ' + str(2 * i))
        plt.ylabel('component ' + str(2 * i + 1))
        plt.colorbar()
        plt.show()

    return pca


# ## portfolio optimization below

class PortfolioOptimization:
    def __init__(self, number_of_asset, number_of_pcs, pca_obj):
        ##set up initial variables
        self.cov = pca_obj.get_covariance()
        self.egval = pca_obj.explained_variance_
        self.egval_mat = np.diag(self.egval)
        self.egvec = pca_obj.components_.T  ## note that the eigenvector output from PCA analysis 
        ## is originally transposed, need to be converted back
        self.asset = number_of_asset
        self.pcs = number_of_pcs
        self.weight = np.array([1 / self.asset] * self.asset).reshape(self.asset, 1)

    def Maximum_Entropy(self, weight):

        ## calculate principle weight
        weight_f = np.dot(self.egvec.T, weight)
        #         assert weight_f.shape == (self.pcs,1)
        ## calculate total portfolio risk
        ## eq(19)_id22
        ## the same as asset-based portfolio risk
        ## Var_port = w^T * Cov * w
        portfolio_risk = weight_f.T @ self.egval_mat @ weight_f
        #         assert portfolio_risk.shape == (1,1)

        ## loop to calculate risk contribution of each factor
        ## the same as PCR in eq(6)_id22
        prc = []
        for i in range(self.pcs):
            prc.append(((np.sqrt(self.egval[i]) * weight_f[i]) ** 2) / portfolio_risk)
        factor_risk = np.array(prc).reshape((self.pcs, 1))
        ## loop to calculate ENB
        ## eq(30)_id22
        risk_sum = 0
        for x in factor_risk:
            risk_sum -= x * np.log(x)
        ENB = -np.exp(risk_sum)

        return ENB

    def get_weight(self):

        guess = self.weight
        bnds = tuple((-1, 1) for x in range(self.asset))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

        opt = scipy.optimize.minimize(self.Maximum_Entropy, guess, method='SLSQP', bounds=bnds, constraints=cons)
        return opt


###Firtly, we explain the whole process of our strategy.
# use a sample
# example = load_data(datetime(1993, 1, 4))
# explain_model_fitting(example)

#%%
# find every monday, which is our re-balance date.
first_day = pd.to_datetime(1993, 1, 4)

monday = [first_day]
one_week = timedelta(weeks=1)
one_day = timedelta(days=1)
for i in range(1, 52 * 26 + 5):
    first_day += one_week
    monday.append(first_day)


#%%
# get risk free rate
interest = pd.read_csv('/Users/apple/PycharmProjects/Quant_method/1990-2018 Fed fund rate.txt', delim_whitespace=True)
interest = pd.to_numeric(interest.value)
interest.index = pd.to_datetime(interest.date.astype(str))

# get sp500
sp500 = pd.read_csv('/Users/apple/PycharmProjects/Quant_method/S&P500Index.csv', delimiter=',',
                    header=0, usecols=['caldt','sprtrn'])
sp500.index = pd.to_datetime(sp500.caldt.astype(str))
#%%

result = dict()
# this step is to simulate return of managed portfolio, but to notice that I ignore the last re-balance date
# because it is the last day of our sample
# I want to add 5 more day than 2-year range  to track the return of holding period
for i in range(52 * 26 + 4):
    start = monday[i]
    time = datetime.strftime(start, '%Y%m%d')
    sample = load_data(start)
    pca = PCA(n_components=5)
    pca.fit(sample[:-5])
    port = PortfolioOptimization(10, 5, pca)
    weight = port.get_weight()

    port_r = []
    port_sharp = []
    port_vol = []
    port_VaR = []

    for k in range(5):
        r = np.dot(sample[-5 + k:], weight)
        port_r.append(r)

        vol = weight * sample[:-5].get_covariance() * weight.T
        port_vol.append(vol)

        present_time = datetime.strftime(start + k * one_day, '%Y-%m-%d')
        sharp = (r - interest.loc[present_time]) / vol
        port_sharp.append(sharp)

    result["r"].append(np.mean(port_r))
    result["vol"].append(np.mean(port_vol))
    result["VaR"].append(np.mean(port_VaR))
    result["sharp_ratio"].append(np.mean(port_sharp))
#%%
# calculate SP500
SP500_return = []
SP500_vol = []
SP500_Sharp =[]
for i in monday:
    time = datetime.strftime(i, '%Y-%m-%d')
    SP500_return.append(sp500.loc[time]['sprtrn'])

    var = np.var(sp500[:[time]])
    SP500_vol.append(var^0.5)

    SP500_Sharp.append((sp500-interest.loc[time])/var)




# compare average return in holding period
plt.plot(monday, result["r"], label='Portfolio_return', color="red")
plt.plot(monday, SP500_return, label='S&P_Return', color="black")
plt.show()

plt.plot()

# compare volatility in holding period
plt.plot(monday, result["vol"], label='Portfolio_vol', color="red")
plt.plot(monday, SP500_vol, label='S&P_vol', color="black")
plt.show()

# compare sharp ratio in holding period
plt.plot(monday, result["sharp"], label='Portfolio_sharp_ratio', color="red")
plt.plot(monday, SP500_Sharp, label='S&P_sharp', color="black")
plt.show()
