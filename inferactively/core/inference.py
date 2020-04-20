#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions for performing variational inference on hidden states 

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from inferactively.distributions import Categorical, Dirichlet
from inferactively.core import softmax, spm_dot, spm_wnorm, spm_cross


def update_posterior_states(A, observation, prior, return_numpy=True, method="FPI", **kwargs):
    """ 
    Update marginal posterior qx using variational inference, with optional selection of a message-passing algorithm
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays) or Categorical]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array, array of arrays (with 1D numpy array entries), int or tuple]:
        The observation (generated by the environment). If single modality, this can be a 1D array 
        (one-hot vector representation) or an int (observation index)
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation) or an int (observation index)
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'return_numpy' [Boolean]:
        True/False flag to determine whether the posterior is returned as a numpy array or a Categorical
    'method' [str]:
        Algorithm used to perform the variational inference. 
        Options: 'FPI' - Fixed point iteration 
                - http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf, slides 13- 18
                - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.221&rep=rep1&type=pdf, slides 24 - 38
                 'VMP  - Variational message passing
                 'MMP' - Marginal message passing
                 'BP'  - Belief propagation
                 'EP'  - Expectation propagation
                 'CV'  - CLuster variation method
    **kwargs: List of keyword/parameter arguments corresponding to parameter values for the respective variational inference algorithm

    Returns
    ----------
    'qx' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational approximation.
    """

    if isinstance(A, Categorical):
        A = A.values

    if A.dtype == "object":
        Nf = A[0].ndim - 1
        Ns = list(A[0].shape[1:])
        Ng = len(A)
        No = []
        for g in range(Ng):
            No.append(A[g].shape[0])
    else:
        Nf = A.ndim - 1
        Ns = list(A.shape[1:])
        Ng = 1
        No = [A.shape[0]]

    if isinstance(observation, Categorical):
        observation = observation.values
        if Ng == 1:
            observation = observation.squeeze()
        else:
            for g in range(Ng):
                observation[g] = observation[g].squeeze()

    if isinstance(observation, (int, np.integer)):
        observation = np.eye(No[0])[observation]

    if isinstance(observation, tuple):
        observation_AoA = np.empty(Ng, dtype=object)
        for g in range(Ng):
            observation_AoA[g] = np.eye(No[g])[observation[g]]

        observation = observation_AoA

    if isinstance(prior, Categorical):
        prior_new = np.empty(Nf, dtype=object)
        if prior.IS_AOA:
            for f in range(Nf):
                prior_new[f] = prior[f].values.squeeze()
        else:
            prior_new[0] = prior.values.squeeze()
        prior = prior_new

    elif prior.dtype != "object":

        prior_new = np.empty(Nf, dtype=object)
        prior_new[0] = prior
        prior = prior_new

    if method == "FPI":
        qx = run_FPI(A, observation, prior, No, Ns, **kwargs)
    if method == "VMP":
        raise NotImplementedError("VMP is not implemented")
    if method == "MMP":
        raise NotImplementedError("MMP is not implemented")
    if method == "BP":
        raise NotImplementedError("BP is not implemented")
    if method == "EP":
        raise NotImplementedError("EP is not implemented")
    if method == "CV":
        raise NotImplementedError("CV is not implemented")

    if return_numpy:
        return qx
    else:
        return Categorical(values=qx)


def run_FPI(A, observation, prior, No, Ns, num_iter=10, dF=1.0, dF_tol=0.001):
    """
    Update marginal posterior beliefs about hidden states
    using variational fixed point iteration (FPI)
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array or array of arrays (with 1D numpy array entries)]:
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation).
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors).
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries)]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'num_iter' [int]:
        Number of variational fixed-point iterations to run.
    'dF' [float]:
        Starting free energy gradient (dF/dt) before updating in the course of gradient descent.
    'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dt), to be checked at each iteration. If 
        dF <= dF_tol, the iterations are halted pre-emptively and the final marginal posterior belief(s) is(are) returned
    Returns
    ----------
    'qx' [numpy 1D array or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational fixed point iteration (mean-field)
    """

    Ng = len(No)
    Nf = len(Ns)

    L = np.ones(tuple(Ns))

    # loop over observation modalities and use mean-field assumption to multiply 'induced posterior' onto
    # a single joint likelihood over hidden factors - of size Ns
    if Ng == 1:
        L *= spm_dot(A, observation, obs_mode=True)
    else:
        for g in range(Ng):
            L *= spm_dot(A[g], observation[g], obs_mode=True)

    L = np.log(L + 1e-16)

    # initialize posterior to flat distribution
    qx = np.empty(Nf, dtype=object)
    for f in range(Nf):
        qx[f] = np.ones(Ns[f]) / Ns[f]

    # F_init = 0
    # dF_Q_init = 0
    # for f in range(Nf):
    #     F_init += -qx[f].dot(np.log(qx[f][:,np.newaxis] + 1e-16)) -qx[f].dot(prior[f][:,np.newaxis])
    #     sum_log_marginal = np.sum( np.log(qx[f] + 1e-16))
    #     sum_qL = np.sum(spm_dot(L, qx, [f]))
    #     sum_prior = np.sum(prior[f])
    #     dF_Q_init += (sum_log_marginal - sum_qL - sum_prior)

    # print('Initial free energy gradient: %.2f\n'%dF_Q_init)
    # print('Initial free energy: %.2f\n'%F_init)

    # initialize the 'previous' free energy (here, the initial free energy)
    F_prev = 0
    for f in range(Nf):
        F_prev += -qx[f].dot(np.log(qx[f][:, np.newaxis] + 1e-16)) - qx[f].dot(
            prior[f][:, np.newaxis]
        )

    if Nf == 1:
        qL = spm_dot(L, qx, [0])
        qx[0] = softmax(qL + prior[0])
        return qx[0]

    else:
        iter_i = 0
        while iter_i < num_iter:

            F = 0  # initialize the variational free energy
            # dF_Q = 0

            # qx_prev = qx.copy()

            for f_loop in range(2):
                if f_loop == 0:
                    factor_order = range(Nf)
                elif f_loop == 1:
                    factor_order = range((Nf - 1), -1, -1)
                for f in factor_order:

                    # get the marginal for hidden state factor f by marginalizing out
                    # other factors (summing them, weighted by their posterior expectation)
                    # qL = spm_dot(L, qx_prev, [f])
                    qL = spm_dot(L, qx, [f])

                    qx[f] = softmax(qL + prior[f])

                    # sum_log_marginal = np.sum( np.log(qx[f] + 1e-16))
                    # sum_qL = np.sum(qL)
                    # sum_prior = np.sum(prior[f])
                    # dF_Q += (sum_log_marginal - sum_qL - sum_prior)
                    # print('Contribution to dF_Q from Marginal %d: %.2f\n'%(f, sum_log_marginal))
                    # print('Contribution to dF_Q from expected log-likelihood %d: %.2f\n'%(f, -sum_qL))
                    # print('Contribution to dF_Q from prior %d: %.2f\n'%(f, -sum_prior))

                    # h_qx= -qx[f].dot(np.log(qx[f][:,np.newaxis] + 1e-16))
                    # xh_qx_px = -qx[f].dot(prior[f][:,np.newaxis])
                    # F += (h_qx + xh_qx_px)
                    # print('Contribution to F from entropy of marginal %d: %.2f\n'%(f, h_qx))
                    # print('Contribution to F from cross entropy with prior %d: %.2f\n'%(f, xh_qx_px))

            for f in range(Nf):
                h_qx = -qx[f].dot(np.log(qx[f][:, np.newaxis] + 1e-16))
                xh_qx_px = -qx[f].dot(prior[f][:, np.newaxis])
                F += h_qx + xh_qx_px

            E_Q_lh = spm_dot(L, qx)[0]
            F -= E_Q_lh
            #  print("Contribution to F from expected log likelihood %d: %.2f\n" % (f, -E_Q_lh))

            # print('Free energy gradient at iteration %d: %.2f\n'%(iter_i,dF_Q))
            # print('Total free energy at iteration %d: %.5f\n'%(iter_i,F))

            dF = np.abs(F_prev - F)
            # print("Free energy difference between iterations: %.5f\n" % (dF))

            F_prev = F

            if dF < dF_tol:
                print("Stopped updating after iteration %d\n" % iter_i)
                break
                # return qx

            iter_i += 1

        return qx


def run_FPI_faster(A, observation, prior, No, Ns, num_iter=10, dF=1.0, dF_tol=0.001):
    """
    Update marginal posterior beliefs about hidden states
    using variational fixed point iteration (FPI). 
    @NOTE (Conor, 26.02.2020):
    This method uses a faster algorithm than the traditional 'spm_dot' approach. Instead of
    separately computing a conditional joint log likelihood of an outcome, under the
    posterior probabilities of a certain marginal, instead all marginals are multiplied into one joint tensor that gives the joint likelihood of 
    an observation under all hidden states, that is then sequentially (and *parallelizably*) marginalized out to get each marginal posterior. 
    This method is less RAM-intensive, admits heavy parallelization, and runs (about 2x) faster.
    @NOTE (Conor, 28.02.2020):
    After further testing, discovered interesting differences  between this version and the original version. It appears that the
    original version (simple 'run_FPI') shows mean-field biases or 'explaining away' effects, whereas this version spreads probabilities more 
    'fairly' among possibilities.
    To summarize: it actually matters what order you do the summing across the joint likelihood tensor. In this verison, all marginals
    are multiplied into the likelihood tensor before summing out, whereas in the previous version, marginals are recursively multiplied and summed out.
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array or array of arrays (with 1D numpy array entries)]:
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation).
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors).
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries)]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'num_iter' [int]:
        Number of variational fixed-point iterations to run.
    'dF' [float]:
        Starting free energy gradient (dF/dQx) before updating in the course of gradient descent.
    'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dQx), to be checked at each iteration. If 
        dF <= dF_tol, the iterations are halted pre-emptively and the final marginal posterior belief(s) is(are) returned
    Returns
    ----------
    'qx' [numpy 1D array or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via 
        variational fixed point iteration (mean-field)
    """

    Ng = len(No)
    Nf = len(Ns)

    L = np.ones(tuple(Ns))

    # loop over observation modalities and multiply 'induced posteriors' onto
    # a single joint likelihood over hidden factors - of size Ns
    if Ng == 1:
        L *= spm_dot(A, observation, obs_mode=True)
    else:
        for g in range(Ng):
            L *= spm_dot(A[g], observation[g], obs_mode=True)

    L = np.log(L + 1e-16)

    # initialize marginal posteriors to flat distribution
    qx = np.empty(Nf, dtype=object)
    for f in range(Nf):
        qx[f] = np.ones(Ns[f]) / Ns[f]

    # initialize the 'previous' free energy (here, the initial free energy)
    F_prev = 0
    for f in range(Nf):
        F_prev += -qx[f].dot(np.log(qx[f][:, np.newaxis] + 1e-16)) - qx[f].dot(
            prior[f][:, np.newaxis]
        )

    # in the trivial case of one hidden state factor, inference doesn't require FPI
    if Nf == 1:
        qL = spm_dot(L, qx, [0])
        qx[0] = softmax(qL + prior[0])
        return qx[0]

    else:
        iter_i = 0
        while iter_i < num_iter:

            F = 0  # initialize the variational free energy

            for f_loop in range(2):
                if f_loop == 0:
                    factor_order = range(Nf)
                elif f_loop == 1:
                    factor_order = range((Nf - 1), -1, -1)

                X = L.copy()  # reset the log likelihood

                for f in factor_order:
                    s = np.ones(np.ndim(X), dtype=int)
                    s[f] = len(qx[f])
                    # X *= qx_prev[f].reshape(tuple(s))
                    X *= qx[f].reshape(tuple(s))
                for f in factor_order:

                    s = np.ones(np.ndim(X), dtype=int)
                    s[f] = len(qx[f])
                    # temp = X * (1.0/qx_prev[f]).reshape(tuple(s)) # divide out the factor we multiplied into X already
                    temp = X * (1.0 / qx[f]).reshape(
                        tuple(s)
                    )  # divide out the factor we multiplied into X already
                    dims2sum = tuple(np.where(np.arange(Nf) != f)[0])
                    qL = np.sum(temp, dims2sum)

                    qx[f] = softmax(qL + prior[f])

                    # h_qx= -qx[f].dot(np.log(qx[f][:,np.newaxis] + 1e-16))
                    # xh_qx_px = -qx[f].dot(prior[f][:,np.newaxis])
                    # F += (h_qx + xh_qx_px)

            for f in range(Nf):
                h_qx = -qx[f].dot(np.log(qx[f][:, np.newaxis] + 1e-16))
                xh_qx_px = -qx[f].dot(prior[f][:, np.newaxis])
                F += h_qx + xh_qx_px

            # in the spm_dot version, you essentially multiply each marginal along the log-likelihood L, and then sum out. Since we've already
            # done the multiplication step above (in the computation of X), all we need to do is sum out the result
            E_Q_lh = spm_dot(L, qx)[0]
            # E_Q_lh = X.sum()
            F -= E_Q_lh
            print("Contribution to F from expected log likelihood %d: %.2f\n" % (f, -E_Q_lh))

            dF = np.abs(F_prev - F)
            print("Free energy difference between iterations: %.5f\n" % (dF))

            F_prev = F

            if dF < dF_tol:
                print("Stopped updating after iteration %d\n" % iter_i)
                break

            iter_i += 1
        return qx