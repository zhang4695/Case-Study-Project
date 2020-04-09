#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fredapi import Fred
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.metrics

# using Fred to import 6 yield series
fred = Fred(api_key='24a0ce6c3a8ef4896af1f82f01397671')
dic = dict()
ticker = ['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10']
for x in ticker:
    dic[x] = fred.get_series(x)
# gs is the dataframe of 6 yield series
gs = pd.DataFrame({x: dic[x] for x in dic.keys()})
gs.dropna(inplace=True)


'''
Pipeline Standardscaler and PCA Together
Here I standardized training data for better performance of PCA then use 
a for loop to implement moving-window PCA. Each time, PCA will get 3 PCs from
previous 24 months data and then project PCs onto the next 12 months.
Then calculate the MSE between projected data and real data and also get explained variance
'''
mse = []
var_score = []
std_pca = make_pipeline(StandardScaler(), PCA(n_components=3))
for i in range(24, len(gs) - 12):
    train = std_pca.fit(gs[i - 24:i])
    test = train.inverse_transform(train.transform(gs[i:i + 12]))
    mse.append(mean_squared_error(gs[i:i + 12], test))
    var_score.append(sklearn.metrics.explained_variance_score(gs[i:i + 12], test))
mse_df = pd.DataFrame({'MSE':mse}, index=gs.index[36:])
var_df = pd.DataFrame({'explained_variance_ratio': var_score}, index=gs.index[36:])

# example of components during 2018-2019
eg_data2018_2019 = StandardScaler().fit_transform(gs[498:])
eg_table = pd.DataFrame(PCA().fit(eg_data2018_2019).components_.T,
                        index=['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10'],
                        columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])

'''
Similarity Testing
Basically I use the pipeline and PCA.score() to see whether this year's model fits 2019.
PCA.score() will produce negative log-likelihood and what we want is the smallest (minimized negative log_like)
'''
yrs = range(1977, 2019)
score = []
for k in yrs:
    sec = gs[(gs.index >= pd.datetime(k, 1, 1)) & (gs.index <= pd.datetime(k, 11, 1))]
    train = std_pca.fit(sec)
    score.append(train.score(gs[511:]))

score_df = pd.DataFrame({'log_like': score}, index=range(1977, 2019))
top_10_likelihood = score_df.sort_values('log_like', ascending=True).head(10)

plt.figure(1)
plt.plot(gs.iloc[:,0], label = 'GS1')
plt.plot(gs.iloc[:,1], label = 'GS2')
plt.plot(gs.iloc[:,2], label = 'GS3')
plt.plot(gs.iloc[:,3], label = 'GS5')
plt.plot(gs.iloc[:,4], label = 'GS7')
plt.plot(gs.iloc[:,5], label = 'GS10')
plt.title('US Treasury Yield')
plt.legend()
# plt.plot(gs, label = ticker)


plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(mse_df)
plt.title('Mean Squared Error')
plt.subplot(2, 1, 2)
plt.plot(var_df)
plt.title('Explained Variance Ratio')
plt.tight_layout()

plt.figure(3)
plt.plot(eg_table.iloc[:,0], label = 'PC1')
plt.plot(eg_table.iloc[:,1], label = 'PC2')
plt.plot(eg_table.iloc[:,2], label = 'PC3')
plt.title('Eigen Matrix of 3 Principal Components(2018-2019)')
plt.legend()

plt.figure(4)
plt.plot(score_df)
plt.title('negative log likelihood of model fitting')

plt.show()



