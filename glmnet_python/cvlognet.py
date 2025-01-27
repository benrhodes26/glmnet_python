# -*- coding: utf-8 -*-
"""
Internal function called by cvglmnet. See also cvglmnet

"""
import numpy as np
import scipy
from glmnetPredict import glmnetPredict
from wtmean import wtmean
from cvcompute import cvcompute

def cvlognet(fit, \
            lambdau, \
            x, \
            y, \
            weights, \
            offset, \
            foldid, \
            ptype, \
            grouped, \
            keep = False):
    
    typenames = {'deviance':'Binomial Deviance', 'mse':'Mean-Squared Error',
                 'rsquared': 'R-squared', 'mae':'Mean Absolute Error', 
                 'auc':'AUC', 'class':'Misclassification Error'}
    if ptype == 'default':
        ptype = 'deviance'
        
    ptypeList = ['mse', 'mae', 'deviance', 'auc', 'class', 'rsquared']    
    if not ptype in ptypeList:
        print('Warning: only ', ptypeList, 'available for binomial models; ''deviance'' used')
        ptype = 'deviance'

    prob_min = 1.0e-5
    prob_max = 1 - prob_min
    nc = y.shape[1]        
    if nc == 1:
        classes, sy = scipy.unique(y, return_inverse = True)
        nc = len(classes)
        indexes = scipy.eye(nc, nc)
        y = indexes[sy, :]
    else:
        classes = scipy.arange(nc) + 1 # 1:nc
        
    N = y.size
    nfolds = scipy.amax(foldid) + 1
    if (N/nfolds < 10) and (type == 'auc'):
        print('Warning: Too few (<10) observations per fold for type.measure=auc in cvlognet')
        print('Warning:     changed to type.measure = deviance. Alternately, use smaller value ')
        print('Warning:     for nfolds')
        ptype = 'deviance'
    
    if (N/nfolds < 3) and grouped:    
        print('Warning: option grouped = False enforced in cvglmnet as there are < 3 observations per fold')
        grouped = False

    result = dict()
    result['name'] = typenames[ptype]

    for i, measure in enumerate([ptype, 'class', 'mse', 'rsquared']):
        name = '' if i == 0 else f'_{measure}'
        _, cvm, cvsd = make_predictions(fit, lambdau, x, y, weights, offset, foldid, measure, 
                                        grouped, prob_min, prob_max, val_mode=False)
        result['trn_cvm' + name] = cvm
        result['trn_cvsd' + name] = cvsd

        predmat, cvm, cvsd = make_predictions(fit, lambdau, x, y, weights, offset, foldid, measure, 
                                            grouped, prob_min, prob_max)
        result['cvm' + name] = cvm
        result['cvsd' + name] = cvsd

    if keep:
        result['fit_preval'] = predmat
        
    return(result)

def make_predictions(fit, lambdau, x, y, weights, offset, foldid, ptype,
                      grouped, prob_min, prob_max, val_mode=True):
    
    nfolds = scipy.amax(foldid) + 1
    assert grouped, "'grouped' must be true"
    assert y.size/nfolds >= 3, f"{y.size/nfolds} is not enough samples per fold. Need at least 3."
    if val_mode:
        predmat = scipy.ones([y.shape[0], lambdau.size])*scipy.NAN
    else:
        predmat = None

    is_offset = not(len(offset) == 0)
    nfolds = scipy.amax(foldid) + 1
    cvraw = scipy.zeros([nfolds, lambdau.size])*scipy.NaN
    cvweights = scipy.zeros([nfolds, 1])
    good = scipy.zeros([nfolds, lambdau.size])
    for i in range(nfolds):
        
        if val_mode:
            which = foldid == i
        else:
            which = foldid != i
        
        wi = weights[which]
        cvweights[i] = scipy.sum(wi, axis = 0)
        fitobj = fit[i].copy()

        nlami = scipy.size(fitobj['lambdau'])
        good[i, 0:nlami] = 1
        
        if is_offset:
            off_sub = offset[which, ]
        else:
            off_sub = scipy.empty([0])

        preds = glmnetPredict(fitobj, x[which, ], scipy.empty([0]), 'response', False, off_sub)
        preds = np.asarray(preds)
        yi = y[which, :]

        if ptype == 'auc':
            for j in range(nlami):
                cvraw[i, j] = auc_mat(yi, preds[:, j], wi)
        else:
            yiwt = scipy.sum(yi, axis = 1, keepdims = True)
            yi = yi/scipy.tile(yiwt, [1, yi.shape[1]])
            yy1 = scipy.tile(yi[:,0:1], [1, lambdau.size])
            yy2 = scipy.tile(yi[:,1:2], [1, lambdau.size])

        if ptype == 'mse':
            cvraw[i] = wtmean((yy2 - preds)**2, wi)
        elif ptype == 'rsquared':
            sserr = wtmean((yy2 - preds)**2, wi)
            sstot = wtmean((yy2 - scipy.mean(yy2, axis=0, keepdims=True))**2, wi)
            cvraw[i] = 1 - (sserr/sstot)
        elif ptype == 'deviance':
            preds = scipy.minimum(scipy.maximum(preds, prob_min), prob_max)
            lp = yy1*scipy.log(1-preds) + yy2*scipy.log(preds)
            ly = scipy.log(yi)
            ly[yi == 0] = 0
            ly = scipy.dot(yi*ly, scipy.array([1.0, 1.0]).reshape([2,1]))
            cvraw[i] = wtmean(2*(scipy.tile(ly, [1, lambdau.size]) - lp), wi)
        elif ptype == 'mae':
            cvraw[i] = wtmean(scipy.absolute(yy1 - (1 - preds)) + scipy.absolute(yy2 - (1 - preds)), wi)
        elif ptype == 'class':
            cvraw[i] = wtmean(yy1*(preds > 0.5) + yy2*(preds <= 0.5), wi)

        if val_mode:
            predmat[which, 0:nlami] = preds

    N = scipy.sum(good, axis = 0)
    cvm = wtmean(cvraw, cvweights)
    sqccv = (cvraw - cvm)**2
    cvsd = scipy.sqrt(wtmean(sqccv, cvweights)/(N-1))

    return predmat, cvm, cvsd


# def make_predictions(fit, lambdau, x, y, weights, offset, foldid, ptype, grouped, prob_min, prob_max, which_set="val"):

#     is_offset = not(len(offset) == 0)
#     predmat = scipy.ones([y.shape[0], lambdau.size])*scipy.NAN               
#     nfolds = scipy.amax(foldid) + 1
#     nlams = []    
#     for i in range(nfolds):
#         which = foldid == i
#         fitobj = fit[i].copy()
#         if is_offset:
#             off_sub = offset[which, ]
#         else:
#             off_sub = scipy.empty([0])
#         preds = glmnetPredict(fitobj, x[which, ], scipy.empty([0]), 'response', False, off_sub)
#         nlami = scipy.size(fit[i]['lambdau'])
#         predmat[which, 0:nlami] = preds
#         nlams.append(nlami)

#     # convert nlams to scipy array
#     nlams = scipy.array(nlams, dtype = scipy.integer)

#     if ptype == 'auc':
#         cvraw = scipy.zeros([nfolds, lambdau.size])*scipy.NaN
#         good = scipy.zeros([nfolds, lambdau.size])
#         for i in range(nfolds):
#             good[i, 0:nlams[i]] = 1
#             which = foldid == i
#             for j in range(nlams[i]):
#                 cvraw[i,j] = auc_mat(y[which,], predmat[which,j], weights[which])
#         N = scipy.sum(good, axis = 0)
#         sweights = scipy.zeros([nfolds, 1])
#         for i in range(nfolds):
#             sweights[i]= scipy.sum(weights[foldid == i], axis = 0)
#         weights = sweights
#     else:
#         ywt = scipy.sum(y, axis = 1, keepdims = True)
#         y = y/scipy.tile(ywt, [1, y.shape[1]])
#         weights = weights*ywt
#         N = y.shape[0] - scipy.sum(scipy.isnan(predmat), axis = 0, keepdims = True)
#         yy1 = scipy.tile(y[:,0:1], [1, lambdau.size])
#         yy2 = scipy.tile(y[:,1:2], [1, lambdau.size])

#     if ptype == 'mse':
#         cvraw = (yy1 - (1 - predmat))**2 + (yy2 - (1 - predmat))**2
#     elif ptype == 'deviance':
#         predmat = scipy.minimum(scipy.maximum(predmat, prob_min), prob_max)
#         lp = yy1*scipy.log(1-predmat) + yy2*scipy.log(predmat)
#         ly = scipy.log(y)
#         ly[y == 0] = 0
#         ly = scipy.dot(y*ly, scipy.array([1.0, 1.0]).reshape([2,1]))
#         cvraw = 2*(scipy.tile(ly, [1, lambdau.size]) - lp)
#     elif ptype == 'mae':
#         cvraw = scipy.absolute(yy1 - (1 - predmat)) + scipy.absolute(yy2 - (1 - predmat))
#     elif ptype == 'class':
#         cvraw = yy1*(predmat > 0.5) + yy2*(predmat <= 0.5)
    
#     if y.size/nfolds < 3 and grouped == True:
#         print('Option grouped=false enforced in cv.glmnet, since < 3 observations per fold')
#         grouped = False
        
#     if grouped == True:
#         cvob = cvcompute(cvraw, weights, foldid, nlams)
#         cvraw = cvob['cvraw']
#         weights = cvob['weights']
#         N = cvob['N']
        
#     cvm = wtmean(cvraw, weights)
#     sqccv = (cvraw - cvm)**2
#     cvsd = scipy.sqrt(wtmean(sqccv, weights)/(N-1))

#     return predmat, cvm, cvsd

# end of cvelnet
#=========================    
#
#=========================    
# Helper functions
#=========================    
def auc_mat(y, prob, weights = None):
    if weights == None or len(weights) == 0:
        weights = scipy.ones([y.shape[0], 1])
    wweights = weights*y
    wweights = wweights.flatten()
    wweights = scipy.reshape(wweights, [1, wweights.size])
    ny= y.shape[0]
    a = scipy.zeros([ny, 1])
    b = scipy.ones([ny, 1])
    yy = scipy.vstack((a, b))
    pprob = scipy.vstack((prob,prob))
    result = auc(yy, pprob, wweights)
    return(result)
#=========================    
def auc(y, prob, w):
    if len(w) == 0:
        mindiff = scipy.amin(scipy.diff(scipy.unique(prob)))
        pert = scipy.random.uniform(0, mindiff/3, prob.size)
        t, rprob = scipy.unique(prob + pert, return_inverse = True)
        n1 = scipy.sum(y, keepdims = True)
        n0 = y.shape[0] - n1
        u = scipy.sum(rprob[y == 1]) - n1*(n1 + 1)/2
        result = u/(n1*n0)
    else:
        op = scipy.argsort(prob)
        y = y[op]
        w = w[op]
        cw = scipy.cumsum(w)
        w1 = w[y == 1]
        cw1 = scipy.cumsum(w1)
        wauc = scipy.sum(w1*(cw[y == 1] - cw1))
        sumw = cw1[-1]
        sumw = sumw*(c1[-1] - sumw)
        result = wauc/sumw
    return(result)    
#=========================    
