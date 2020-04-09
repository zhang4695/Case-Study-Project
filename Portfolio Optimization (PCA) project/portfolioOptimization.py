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

class portfolio_optimization():
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
        bnds = tuple((0, 1) for x in range(self.asset))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

        opt = scipy.optimize.minimize(self.Maximum_Entropy, guess, method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x





sample = load_data(start)
pca = PCA(n_components=6)
pca.fit(sample[:-5])
port = portfolio_optimization(20, 6, pca)
weight = port.get_weight()


