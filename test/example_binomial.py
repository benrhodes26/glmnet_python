# Import relevant modules and setup for calling glmnet

import os
import sys
sys.path.append('../test')
sys.path.append('../lib')

import scipy
import importlib
import matplotlib.pyplot as plt

import glmnet_python as glmnet
import glmnetPlot
import glmnetPrint
import glmnetCoef
import glmnetPredict

import cvglmnet
import cvglmnetCoef
import cvglmnetPlot
import cvglmnetPredict

importlib.reload(glmnet)
importlib.reload(glmnetPlot)    
importlib.reload(glmnetPrint)
importlib.reload(glmnetCoef)    
importlib.reload(glmnetPredict)

importlib.reload(cvglmnet)    
importlib.reload(cvglmnetCoef)
importlib.reload(cvglmnetPlot)
importlib.reload(cvglmnetPredict)

# get the parent directory of current directory
par = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
baseDataDir = os.path.join(par, 'data')

# load data
x = scipy.loadtxt(os.path.join(baseDataDir, 'BinomialExampleX.dat'), dtype = scipy.float64, delimiter = ',')
y = scipy.loadtxt(os.path.join(baseDataDir, 'BinomialExampleY.dat'), dtype = scipy.float64)

# call glmnet
fit = glmnet.glmnet(x = x.copy(), y = y.copy(), family = 'binomial')
                    
glmnetPlot.glmnetPlot(fit, xvar = 'dev', label = True);

glmnetPredict.glmnetPredict(fit, newx = x[0:5,], ptype='class', s = scipy.array([0.05, 0.01]))

cvfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial', ptype = 'class')

plt.figure()
cvglmnetPlot.cvglmnetPlot(cvfit)

cvfit['lambda_min']
cvfit['lambda_1se']

cvglmnetCoef.cvglmnetCoef(cvfit, s = 'lambda_min')

cvglmnetPredict.cvglmnetPredict(cvfit, newx = x[0:10, ], s = 'lambda_min', ptype = 'class')

fig, ax = plt.subplots(1,1)
ax.plot(cvfit['trn_cvm'], label='trn')
ax.plot(cvfit['cvm'], label='val')
fig.legend()
fig.show()
fig.savefig(os.path.join(baseDataDir, "tst_cvplot"))