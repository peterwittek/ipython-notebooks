# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:54:09 2016
"""
from __future__ import print_function, division
import cvxopt
import itertools
import math
import multiprocessing
import numpy as np
import numpy.linalg
import os
import picos
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from functools import partial

# We define the Pauli matrices as a global constant
Pauli = [np.array([[0, 1], [1, 0]]),     # sigma_x
         np.array([[0, -1j], [1j, 0]]),  # sigma_y
         np.array([[1, 0], [0, -1]])]    # sigma_z


def basis(k, dim=3):
    """Returns a basis vector |k> in the computational basis.

    :param k: The index of the basis vector.
    :type k: int.
    :param dim: Optional parameter to specify the dimension of the Hilbert
                space. Default value is 3.
    :type dim: int.

    :returns: :class:`numpy.array`
    """
    qutrit = np.zeros((dim, 1), dtype=np.float64)
    qutrit[k] = 1
    return qutrit


def check_ranks(M, tolerance=1e-10):
    return [numpy.linalg.matrix_rank(Mi, tol=tolerance) for Mi in M]


def _check_qubit_sdp(K, solver=None):
    """Verifies whether a qubit POVM is projective-simulable and checks for the
    visibility that would make it simulable. It returns a visibility value,
    which is one if the POVM is simulable.

    :param K: The POVM.
    :type K: list of :class:`numpy.array`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   or "cvxopt". The default is `None`, which triggers
                   autodetect.
    :type solver: str.

    :returns: float
    """
    # Calculate the noise for each effect.
    noise = [np.trace(Ki)*np.eye(2)/2 for Ki in K]
    # Instantiate the convex optimization class
    problem = picos.Problem(verbose=0)
    # Declare the variables of the SDP problem
    q = problem.add_variable('q', 6, lower=[0.0] * 6)
    P = [[problem.add_variable("P_%s%s" % (i, j), (2, 2), vtype='hermitian')
          for j in range(2)] for i in range(6)]
    # Add SDP constraints
    for i in range(6):
        for j in range(2):
            problem.add_constraint(P[i][j] >> 0)
    # q_i add up to one
    problem.add_constraint(sum([q[i] for i in range(6)]) == 1)
    # Pairwise these operators add up to a scaled identiy operator
    for i in range(6):
        problem.add_constraint(P[i][0] + P[i][1] == q[i]*np.eye(2))
    # If the user asked for optimal visibility, we turn the SDP problem to an
    # an actual optimization, otherwise, it is a feasibility problem
    # t is the visibility, which is lower bounded by 0.
    t = problem.add_variable('t', 1, lower=0.0)
    # Techinically, it is upper bounded by 1.0, but we have to specify it
    # in a constraint. It is probably a bug in PICOS.
    problem.add_constraint(t <= 1.0)
    problem.add_constraint(t*K[0] + (1 - t)*noise[0] ==
                           P[0][0] + P[1][0] + P[2][0])
    problem.add_constraint(t*K[1] + (1 - t)*noise[1] ==
                           P[0][1] + P[3][0] + P[4][0])
    problem.add_constraint(t*K[2] + (1 - t)*noise[2] ==
                           P[1][1] + P[3][1] + P[5][0])
    problem.add_constraint(t*K[3] + (1 - t)*noise[3] ==
                           P[2][1] + P[4][1] + P[5][1])
    problem.set_objective('max', t)

    problem.solve(solver=solver)
    if problem.status.count("optimal") > 0:
        obj = problem.obj_value()
    else:
        obj = None
    return obj


def _check_qutrit_3out_sdp(K, solver=None):
    """Verifies whether a qutrit POVM is 3-outcome-simulable and checks for
    the visibility that would make it simulable. It returns a visibility value,
    which is one if the POVM is simulable.

    :param K: The POVM.
    :type K: list of :class:`numpy.array`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   or "cvxopt". The default is `None`, which triggers
                   autodetect.
    :type solver: str.

    :returns: float
    """
    # Exact same logic as before, except that the POVM might have a different
    # of effects, and we have a lot more ways of simulating a POVM.
    n_outcomes = len(K)
    problem = picos.Problem(verbose=0)
    n_three_sets = n_choose_r(n_outcomes, 3)
    P = [[problem.add_variable("P^%s_%s" % (i, j), (3, 3), vtype='hermitian')
          for j in range(3)] for i in range(n_three_sets)]
    p = problem.add_variable("p", n_three_sets, lower=[0.0] * n_three_sets)
    problem.add_list_of_constraints([sum(P[i][j] for j in range(3)) ==
                                     p[i]*np.eye(3)
                                     for i in range(n_three_sets)],
                                    'i', '0...%s' % (n_three_sets-1))
    problem.add_list_of_constraints([P[i][j] >> 0 for i in range(n_three_sets)
                                     for j in range(3)],
                                    'ij', '0...%s, 0...%s' %
                                    (n_three_sets-1, 2))
    problem.add_constraint(sum(p[i] for i in range(n_three_sets)) == 1)
    t = problem.add_variable('t', 1, lower=0.0)
    problem.set_objective('max', t)
    problem.add_constraint(t <= 1.0)
    problem.add_constraint(t >= 0.0)
    noise = [np.trace(Ki).real*np.eye(3)/3 for Ki in K]
    for k in range(n_outcomes):
        sP = sum(P[i][c.index(k)]
                 for i, c in
                 enumerate(itertools.combinations(range(n_outcomes), 3))
                 if c.count(k) > 0)
        problem.add_constraint(sP == t*K[k] + (1-t)*noise[k])
    problem.solve(solver=solver)
    if 'optimal' not in problem.status:
        raise Exception(problem.status)
    if problem.status.count("optimal") > 0:
        obj = problem.obj_value()
    else:
        obj = None
    return obj


def _check_qutrit_proj_sdp(K, solver=None):
    """Verifies whether a qutrit POVM is projective-simulable and checks for
    the visibility that would make it simulable. It returns a visibility value,
    which is one if the POVM is simulable.

    :param K: The POVM.
    :type K: list of :class:`numpy.array`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   or "cvxopt". The default is `None`, which triggers
                   autodetect.
    :type solver: str.

    :returns: float
    """
    # Exact same logic as before, except that the POVM might have a different
    # of effects, and we have a lot more ways of simulating a POVM.
    n_outcomes = len(K)
    problem = picos.Problem(verbose=0)
    n_three_sets = n_choose_r(n_outcomes, 3)
    P = [[problem.add_variable("P^%s_%s" % (i, j), (3, 3), vtype='hermitian')
          for j in range(3)] for i in range(n_three_sets)]
    p = problem.add_variable("p", n_three_sets, lower=[0.0] * n_three_sets)
    n_two_sets = n_choose_r(n_outcomes, 2)
    R = [[problem.add_variable("R^%s_%s" % (i, j), (3, 3), vtype='hermitian')
          for j in range(2)] for i in range(n_two_sets)]
    r = problem.add_variable("r", n_two_sets, lower=[0.0] * n_two_sets)
    problem.add_list_of_constraints([sum(P[i][j] for j in range(3)) ==
                                     p[i]*np.eye(3)
                                     for i in range(n_three_sets)],
                                    'i', '0...%s' % (n_three_sets-1))
    problem.add_list_of_constraints([P[i][j] >> 0 for i in range(n_three_sets)
                                     for j in range(3)],
                                    'ij', '0...%s, 0...%s' %
                                    (n_three_sets-1, 2))
    problem.add_list_of_constraints([picos.trace(P[i][j]) == p[i]
                                     for i in range(n_three_sets)
                                     for j in range(3)],
                                    'ij', '0...%s, 0...%s' %
                                    (n_three_sets-1, 2))
    problem.add_list_of_constraints([R[i][j] >> 0 for i in range(n_two_sets)
                                     for j in range(2)],
                                    'i_j', '0...%s, 0...%s' %
                                    (n_two_sets-1, 1))
    problem.add_list_of_constraints([sum(R[i][j] for j in range(2)) ==
                                     r[i]*np.eye(3)
                                     for i in range(n_two_sets)],
                                    'i', '0...%s' % (n_two_sets-1))

    problem.add_constraint(sum(p[i] for i in range(n_three_sets)) +
                           sum(r[i] for i in range(n_two_sets)) == 1)
    t = problem.add_variable('t', 1, lower=0.0)
    problem.set_objective('max', t)
    problem.add_constraint(t <= 1.0)
    problem.add_constraint(t >= 0.0)
    noise = [np.trace(Ki).real*np.eye(3)/3 for Ki in K]
    for k in range(n_outcomes):
        sP = sum(P[i][c.index(k)]
                 for i, c in
                 enumerate(itertools.combinations(range(n_outcomes), 3))
                 if c.count(k) > 0)
        sR = sum(R[i][c.index(k)]
                 for i, c in
                 enumerate(itertools.combinations(range(n_outcomes), 2))
                 if c.count(k) > 0)
        problem.add_constraint(sP + sR == t*K[k] + (1 - t)*noise[k])
    problem.solve(solver=solver)
    if problem.status.count("optimal") > 0:
        obj = problem.obj_value()
    else:
        obj = None
    return obj


def complex_cross_product(v, w):
    """Returns |v><w|

    :returns: :class:`numpy.array`
    """
    qutrit = np.array([np.cross(w.conj().T[0], v.conj().T[0])]).T
    return qutrit/np.linalg.norm(qutrit)


def create_qubit_povm_from_vector(v):
    K = [v[0]*np.eye(2) + v[0]*Pauli[0]]
    K += [v[1]*np.eye(2) + v[2]*Pauli[0] + v[3]*Pauli[1]]
    K += [v[4]*np.eye(2) + v[5]*Pauli[0] +
          v[6]*Pauli[1] + v[7]*Pauli[2]]
    K += [np.eye(2) - K[0] - K[1] - K[2]]
    return K


def create_covariant_povm_from_vector(v):
    """Given a 8-dimensional real vector describing a covariant qutrit
    measurement, it returns the list of effects.

    :returns: list of :class:`numpy.array`
    """
    w = np.cos(2*np.pi/3) + 1j*np.sin(2*np.pi/3)
    x = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    D = [[], [], []]
    for j in range(3):
        for k in range(3):
            D[j].append(np.matrix((w**(j*k/2))*sum(
                w**(j*m)*x[:, np.mod(k + m, 3)]*dag(x[:, m]) for m in
                    range(3))))
    eff = np.matrix([[v[0], v[1] + v[2]*1j, v[3] + v[4]*1j],
                     [v[1] - v[2]*1j, v[5], v[6] + v[7]*1j],
                     [v[3] - v[4]*1j, v[6] - v[7]*1j, 1/3 - v[0] - v[5]]])
    M = []
    for j in range(3):
        for k in range(3):
            M.append(D[j][k]*eff*dag(D[j][k]))
    return M


def dag(v):
    """Return v^\dagger

    :returns: :class:`numpy.array`
    """
    return v.conj().T


def _decompose333To233(M):
    """ The first step in the decomposition of trace-one qutrit POVMs into
    projective ones. Since the output of each step is the input of the next
    one, we decrease the precision of the SDPs in each step to avoid error
    propagation.

    """
    _, B = numpy.linalg.eigh(M[0])
    P = np.array([B[:, i:i +1]*dag(B[:, i:i +1]) for i in range(len(M))])
    problem = picos.Problem(verbose=0)
    t = problem.add_variable('t', 1, lower=0.0)
    problem.set_objective('max', t)
    problem.add_list_of_constraints([-t*P[i] + M[i] >> 0
                                     for i in range(len(M))])
    problem.set_option('mosek_params', solverparameters(1e-10))
    problem.solve(solver='mosek')
    if 'optimal' not in problem.status and 'feasible' not in problem.status:
        raise Exception(problem.status)
    t1 = problem.obj_value()
    N = np.array([(M[i] - t1*P[i])/(1-t1) for i in range(len(M))])
    print([numpy.linalg.eigvalsh(M[i] - t1*P[i]) for i in range(len(M))])
    return P, N, t1


def _decompose233To223(M):
    l = check_ranks(M, 1e-09)
    if l[0] == 2:
        k = 0
    elif l[1] == 2:
        k = 1
    else:
        k = 2
    _, B = numpy.linalg.eigh(M[k])
    PP = np.array([B[:, i:i + 1]*dag(B[:, i:i + 1]) for i in range(len(M))])
    P = []
    if k == 0:
        P.append(PP[2])
        P.append(PP[0])
        P.append(PP[1])
    elif k == 1:
        P.append(PP[0])
        P.append(PP[2])
        P.append(PP[1])
    else:
        for i in range(3):
            P.append(PP[i])
    P = np.array(P)
    problem = picos.Problem(verbose=0)
    t = problem.add_variable('t', 1, lower=0.0)
    problem.set_objective('max', t)
    problem.add_list_of_constraints([-t*P[i] + M[i] >> 0
                                     for i in range(len(M))])
    problem.set_option('mosek_params', solverparameters(1e-08))
    problem.solve(solver=solver)
    if not 'optimal' in problem.status:
        raise Exception(problem.status)
    t1 = problem.obj_value()
    N = np.array([(M[i] - t1*P[i])/(1-t1) for i in range(len(M))])
    print([numpy.linalg.eigvalsh(M[i] - t1*P[i]) for i in range(len(M))])
    return P, N, t1


def _decompose223To222(M):
    # Findind the rank-2 effects M[k1], M[k2]
    r = check_ranks(M, 1e-07)
    if r[0] == 3:
        k0 = 0
        k1 = 1
        k2 = 2
    elif r[1] == 3:
        k0 = 1
        k1 = 0
        k2 = 2
    elif r[2] == 3:
        k0 = 2
        k1 = 0
        k2 = 1
    else:
        raise Exception('The POVM is not (2, 2, 3)-type')
    # Finding the eigenvectors associated to the non-null eigenvalues
    A1, B1 = numpy.linalg.eigh(M[k1])
    v1 = B1[:, 1:2]
    v2 = B1[:, 2:3]
    A2, B2 = numpy.linalg.eigh(M[k2])
    u1 = B2[:, 1:2]
    u2 = B2[:, 2:3]
    # Finding the intersection of the supports of M[k1] and M[k2]
    n1 = dag(numpy.cross(v1, v2, axisa=0, axisb=0))
    n2 = dag(numpy.cross(u1, u2, axisa=0, axisb=0))
    psi1 = dag(numpy.cross(n1, n2, axisa=0, axisb=0))
    # Finding an orthogonal vector to psi1 in the support of M[k2]
    psi2 = dag(numpy.cross(psi1, n2, axisa=0, axisb=0))
    # Finding the projector that completes the POVM
    psi0 = dag(numpy.cross(psi1, psi2, axisa=0, axisb=0))
    psi0 = numpy.matrix(psi0/numpy.linalg.norm(psi0))
    psi1 = numpy.matrix(psi1/numpy.linalg.norm(psi1))
    psi2 = numpy.matrix(psi2/numpy.linalg.norm(psi2))
    Psi0 = psi0*dag(psi0)
    Psi1 = psi1*dag(psi1)
    Psi2 = psi2*dag(psi2)
    P = numpy.zeros((3, 3, 3), dtype=complex)
    P[k0] = Psi0
    P[k1] = Psi1
    P[k2] = Psi2
    # Finding the decomposition
    problem = picos.Problem(verbose=0)
    t = problem.add_variable('t', 1)
    problem.set_objective('max', t)
    problem.add_constraint(t >= 0)
    problem.add_list_of_constraints([-t*P[i] + M[i] >> 0
                                     for i in range(len(M))])
    problem.set_option('mosek_params', solverparameters(1e-06))
    problem.solve(solver='mosek')
    if 'optimal' not in problem.status:
        raise Exception(problem.status)
    t1 = problem.obj_value()
    N = numpy.array([(M[i] - t1*P[i])/(1 - t1) for i in range(len(M))])
    print([numpy.linalg.eigvalsh(M[i] - t1*P[i]) for i in range(len(M))])
    return P, N, t1


def _decompose222To122(M):
    """Virtually the last step of the decomposition, since POVMs of type
    (1, 2, 2) are equivalent to a 2-outcome qubit POVM, and therefore
    projective-simulable. Also, in practice we see that the outputs are
    tipically projective POVMs.

    """

    # Finding the eigenvectors associated to the non-null eigenvalues
    A0, B0 = numpy.linalg.eigh(M[0])
    v1 = B0[:, 1:2]
    v2 = B0[:, 2:3]
    A1, B1 = numpy.linalg.eigh(M[1])
    u1 = B1[:, 1:2]
    u2 = B1[:, 2:3]
    A2, B2 = numpy.linalg.eigh(M[2])
    w1 = B2[:, 1:2]
    w2 = B2[:, 2:3]
    # Finding the intersection of the supports
    n0 = dag(numpy.cross(v1, v2, axisa=0, axisb=0))
    n1 = dag(numpy.cross(u1, u2, axisa=0, axisb=0))
    n2 = dag(numpy.cross(w1, w2, axisa=0, axisb=0))
    psi01 = dag(numpy.cross(n0, n1, axisa=0, axisb=0))
    psi02 = dag(numpy.cross(n0, n2, axisa=0, axisb=0))
    psi12 = dag(numpy.cross(n1, n2, axisa=0, axisb=0))
    # Constructing the perturbation
    psi01 = numpy.matrix(psi01/numpy.linalg.norm(psi01))
    psi02 = numpy.matrix(psi02/numpy.linalg.norm(psi02))
    psi12 = numpy.matrix(psi12/numpy.linalg.norm(psi12))
    Psi01 = psi01*dag(psi01)
    Psi02 = psi02*dag(psi02)
    Psi12 = psi12*dag(psi12)
    X = [Psi01 - Psi02, Psi12 - Psi01, Psi02 - Psi12]
    # Finding the decomposition
    cX = [cvxopt.matrix(X[xi]) for xi in range(3)]
    problem = picos.Problem(verbose=0)
    t = problem.add_variable('t', 1)
    problem.add_constraint(t >= 0)
    problem.set_objective('max', t)
    problem.add_list_of_constraints([t*cX[i] + cvxopt.matrix(M[i]) >> 0
                                     for i in range(3)])
    problem.set_option('mosek_params', solverparameters(1e-04))
    problem.solve(solver='mosek')
    if 'optimal' not in problem.status:
        raise Exception(problem.status)
    t1 = problem.obj_value()
    M1 = numpy.array([(M[i] + t1*X[i]) for i in range(3)])
    problem = picos.Problem(verbose=0)
    t = problem.add_variable('t', 1)
    problem.set_objective('max', t)
    problem.add_constraint(t >= 0)
    problem.add_list_of_constraints([-t*cX[i] + cvxopt.matrix(M[i]) >> 0
                                     for i in range(3)])
    problem.set_option('mosek_params', solverparameters(1e-04))
    problem.solve(solver='mosek')
    if 'optimal' not in problem.status:
        raise Exception(problem.status)
    t2 = problem.obj_value()
    M2 = numpy.array([(M[i] - t2*X[i]) for i in range(3)])
    print([numpy.linalg.eigvalsh(M1[i]) for i in range(len(M))])
    print([numpy.linalg.eigvalsh(M2[i]) for i in range(len(M))])
    return M1, M2, t2/(t1 + t2), t1/(t1 + t2)


def decomposePovmToProjective(M):
    coeff, proj_meas = [], []
    if sum(check_ranks(M, 1e-16)[i] for i in range(3)) == 9:
        P1, M1, l1 = _decompose333To233(M)
        coeff += [l1, 1-l1]
        proj_meas += [P1]
    else:
        P1 = np.zeros((3, 3, 3), dtype=complex)
        M1 = M
        coeff += [0, 1]
        proj_meas += [P1]
    # M = coeff[0]*proj_meas[0] + coeff[1]*M1
    if sum(check_ranks(M1, 1e-09)[i] for i in range(3)) == 8:
        P2, M2, l2 = _decompose233To223(M1)
        coeff += [l2, 1-l2]
        proj_meas += [P2]
    else:
        P2 = np.zeros((3, 3, 3), dtype=complex)
        M2 = M1
        coeff += [0, 1]
        proj_meas += [P2]
    # M1 = coeff[2]*proj_meas[1] + coeff[3]*M2
    if sum(check_ranks(M2, 1e-07)[i] for i in range(3)) == 7:
        P3, M3, l3 = _decompose223To222(M2)
        coeff += [l3, 1-l3]
        proj_meas += [P3]
    else:
        P3 = np.zeros((3, 3, 3), dtype=complex)
        M3 = M2
        l3 = 0
        coeff += [l3, 1-l3]
        proj_meas += [P3]
    # M2 = coeff[4]*proj_meas[2] + coeff[5]*M3
    if max(check_ranks(M3, 1e-06)) > 1:
        M4, M5, l4, l5 = _decompose222To122(M3)
        coeff += [l4, l5]
        proj_meas += [M4, M5]
    else:
        M4 = np.zeros((3, 3, 3), dtype=complex)
        M5 = M3
        l4, l5 = 0, 1
        coeff += [l4, l5]
        proj_meas += [M4, M5]
    # M3 = coeff[6]*proj_meas[3] + coeff[7]*proj_meas[4]
    return coeff, proj_meas


def enumerate_vertices(inequalities, method="plrs", verbose=0, rational=False):
    """This is the main function for enumerating vertices given a set of
    inequalities. It is a wrapper to be able to call different enumeration
    algorithms via the same interface.

    :param inequalities: The inequalities that define the polytope of
                         quasi-POVMs. They are given as [b|A], where b is the
                         constant vector, and A is the coefficient matrix.
    :type inequalities: :class:`numpy.array`.
    :param method: Optional parameter to specify the algorithm. Possible
                   values are "lrs", "plrs" (default), and "cdd".
    :type method: str.
    :param verbose: Optional parameter for controlling verbosity.
    :type verbose: int.


    :returns: :class:`numpy.array`.
    """
    if method == "lrs":
        return _run_plrs(inequalities, solverexecutable="lrs", verbose=verbose,
                         rational=rational)
    elif method == "plrs":
        return _run_plrs(inequalities, verbose=verbose, rational=rational)
    elif method == "cdd":
        return _run_cdd(inequalities, verbose=verbose, rational=rational)
    else:
        raise Exception("Unknown method")


def find_best_shrinking_factor(ext, dim, solver=None, parallel=True):
    """Universal optimization function that iterates over a given set of
    extreme points, and returns a matching list of shrinking factors. It is
    universal because it works for both qubits and qutrits. Ideally, it runs
    in parallel to speed up the search.

    :param ext: The set of extreme points.
    :type ext: :class:`numpy.array`.
    :param dim: Whether we optimize over qubit or qutrit quasi-POVMs
    :type dim: int.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   or "cvxopt". The default is `None`, which triggers
                   autodetect.
    :type solver: str.
    :param parallel: Optional parameter for disabling parallel computations.
                     In Python 3, a deadlock may occur for unknown reasons.
    :type parallel: bool.

    :returns: :class:`numpy.array`.
    """

    func = partial(_solve_sdp, dim=dim, solver=solver)

    alphas = np.ones(ext.shape[0])
    # We do an extremely clever application of the best practices of functional
    # programming: we apply a map operator with the specified function over
    # the extreme points provided.
    if parallel:
        import multiprocessing
        pool = multiprocessing.Pool()
        iter_ = pool.imap(func, ((i, v) for i, v in enumerate(ext)), 4)
    else:
        iter_ = map(func, ((i, v) for i, v in enumerate(ext)))

    # Here we simple execute the iterators returned by the map operator.
    for row_index, optimum in iter_:
        alphas[row_index] = optimum
        alpha = min(alphas)
        sys.stdout.write("\r\x1b[KCurrent alpha: {:.6} (done: {:.2%})       "
                         .format(alpha, (row_index+1)/ext.shape[0]))
        sys.stdout.flush()
    # Housekeeping if we had parallel code running
    if parallel:
        pool.close()
        pool.join()
    return alphas


def get_random_qutrit():
    """Returns a random qutrit POVM.

    :returns: :class:`numpy.array`.
    """
    qutrit = np.array([[(random.random()-0.5)*2 + 2j*(random.random()-0.5)],
                       [(random.random()-0.5)*2 + 2j*(random.random()-0.5)],
                       [(random.random()-0.5)*2 + 2j*(random.random()-0.5)]])
    return qutrit/np.linalg.norm(qutrit)


def get_visibility(K, solver=None):
    """It returns a visibility value for a qubit or qutrit POVM, which is one
    if the POVM is simulable.

    :param K: The POVM.
    :type K: list of :class:`numpy.array`.
    :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                   or "cvxopt". The default is `None`, which triggers
                   autodetect.
    :type solver: str.

    :returns: float
    """
    if K[0].shape == (2, 2):
        return _check_qubit_sdp(K, solver)
    elif K[0].shape == (3, 3):
        return _check_qutrit_proj_sdp(K, solver)
    else:
        raise Exception("Not implemented for %d dimensions" % K[0].shape[0])


def n_choose_r(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def _read_ext(filename, rational=False):
    """Helper function for using the lrs/plrs algorithms. It is only called by
    `run_plrs`.
    """
    f = open(filename, 'r')
    result = []
    line = ""
    while line.find("begin") < 0:
        line = f.readline()
    line = f.readline()
    while True:
        line = f.readline()
        if line.find("end") > -1:
            break
        point = []
        i = 0
        for number in line.split(" "):
            if number != "" and number != "\n":
                if rational:
                    point.append(Fraction(number))

                else:
                    point.append(float(Fraction(number)))
                i += 1
        result.append(point)
    return np.array(result)


def _run_cdd(inequalities, verbose=0, rational=False):
    """Helper function called by `enumerate_vertices` to run the CDD algorithm
    to enumerate vertices.
    """
    import cdd
    if rational:
        mat = cdd.Matrix(inequalities, number_type='fraction')
    else:
        mat = cdd.Matrix(inequalities)
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    upper_limit = ext.row_size - len(ext.lin_set)
    result = []
    for row_index in range(upper_limit):
        result.append(np.array(ext[row_index]))
    result = np.array(result)
    if sum(result[:, 0]) != ext.row_size:
        print("Warning: rays in the solution!")
    return result


def _run_plrs(inequalities=None, file_in=None, file_out=None,
              solverexecutable="plrs", threads=None, verbose=0,
              rational=False):
    """Helper function called by `enumerate_vertices` to run the lrs/plrs
    algorithm to enumerate vertices.
    """
    tempfile_ = tempfile.TemporaryFile()
    tmp_filename = tempfile_.name
    tempfile_.close()

    if file_in is None:
        file_in_ = str(tmp_filename) + ".ine"
        _write_to_ine(file_in_, inequalities)
    else:
        file_in_ = file_in
    if file_out is None:
        file_out_ = str(tmp_filename) + ".plrs"
    else:
        file_out_ = file_out
    command_line = [solverexecutable, file_in_, file_out_]
    if solverexecutable.find("plrs") > -1:
        if threads is None:
            threads = multiprocessing.cpu_count()
        command_line += ["-mt", str(threads)]
    try:
        if verbose == 0:
            with open(os.devnull, "w") as fnull:
                subprocess.call(command_line, stdout=fnull, stderr=fnull)
        else:
            subprocess.call(command_line)
    except:
        if file_in is None and verbose < 2:
            os.remove(file_in_)
        if file_out is None and verbose < 2:
            os.remove(file_out_)
        raise Exception
    if file_in is None and verbose < 2:
        os.remove(file_in_)
    if file_out is None and verbose < 2:
        result = _read_ext(file_out_, rational)
        os.remove(file_out_)
    else:
        result = _read_ext(file_out_, rational)
    return result


def _solve_sdp(args, dim, solver):
    """The function to check the shrinking factor of an extreme point.It
    returns the index of the extreme point and the matching shrinking factor.
    This is the function to feed to `find_best_shrinking_factor`, along with a
    set of extreme points.
    """
    povm_vector = args[1][1:]
    if dim == 2:
        K = create_qubit_povm_from_vector(povm_vector)
        obj = _check_qubit_sdp(K, solver)
    elif dim == 3:
        K = create_covariant_povm_from_vector(povm_vector)
        obj = _check_qutrit_proj_sdp(K, solver)
    if obj is not None:
        return args[0], obj
    else:
        return args[0], 1.0


def solverparameters(precision):
    return {'intpnt_co_tol_rel_gap': precision,
            'intpnt_co_tol_mu_red': precision,
            'intpnt_nl_tol_rel_gap': precision,
            'intpnt_nl_tol_mu_red': precision,
            'intpnt_tol_rel_gap': precision,
            'intpnt_tol_mu_red': precision,
            'intpnt_co_tol_dfeas': precision,
            'intpnt_co_tol_infeas': precision,
            'intpnt_co_tol_pfeas': precision}


def truncatedicosahedron(ratio):

    phi = 1+np.sqrt(5)/2
    r = np.zeros((60, 3))
    r[0] = [3*phi, 0, 1]
    r[1] = -r[0]
    r[2] = [3*phi, 0, -1]
    r[3] = -r[2]

    r[4] = [1+2*phi, phi, 2]
    r[5] = -r[4]
    r[6] = [1+2*phi, -phi, 2]
    r[7] = -r[6]
    r[8] = [1+2*phi, phi, -2]
    r[9] = -r[8]
    r[10] = [1+2*phi, -phi, -2]
    r[11] = -r[10]

    r[12] = [2+phi, 2*phi, 1]
    r[13] = -r[12]
    r[14] = [2+phi, 2*phi, -1]
    r[15] = -r[14]
    r[16] = [2+phi, -2*phi, 1]
    r[17] = -r[16]
    r[18] = [2+phi, -2*phi, -1]
    r[19] = -r[18]

    r[20] = [1, 3*phi, 0]
    r[21] = -r[20]
    r[22] = [1, -3*phi, 0]
    r[23] = -r[22]

    r[24] = [2, 1+2*phi, phi]
    r[25] = -r[24]
    r[26] = [2, 1+2*phi, -phi]
    r[27] = -r[26]
    r[28] = [-2, 1+2*phi, phi]
    r[29] = -r[28]
    r[30] = [-2, 1+2*phi, -phi]
    r[31] = -r[30]

    r[32] = [1, 2+phi, 2*phi]
    r[33] = -r[32]
    r[34] = [1, 2+phi, -2*phi]
    r[35] = -r[34]
    r[36] = [-1, 2+phi, 2*phi]
    r[37] = -r[36]
    r[38] = [-1, 2+phi, -2*phi]
    r[39] = -r[38]

    r[40] = [0, 1, 3*phi]
    r[41] = -r[40]
    r[42] = [0, -1, 3*phi]
    r[43] = -r[42]

    r[44] = [phi, 2, 1+2*phi]
    r[45] = -r[44]
    r[46] = [phi, -2, 1+2*phi]
    r[47] = -r[46]
    r[48] = [-phi, 2, 1+2*phi]
    r[49] = -r[48]
    r[50] = [-phi, -2, 1+2*phi]
    r[51] = -r[50]

    r[52] = [2*phi, 1, 2+phi]
    r[53] = -r[52]
    r[54] = [2*phi, -1, 2+phi]
    r[55] = -r[54]
    r[56] = [-2*phi, 1, 2+phi]
    r[57] = -r[56]
    r[58] = [-2*phi, -1, 2+phi]
    r[59] = -r[58]

    for i in range(60):
        r[i] = ratio*r[i]/np.linalg.norm(r[i])
    return r


def _write_to_ine(filename, inequalities):
    """Helper function for using the lrs/plrs algorithms. It is only called by
    `run_plrs`.
    """

    f = open(filename, 'w')
    f.write("begin\n")
    f.write(str(inequalities.shape[0]) + " " + str(inequalities.shape[1]))
    f.write(" rational\n")
    for inequality in inequalities:
        for number in inequality:
            n = Fraction(number)
            if n.denominator == 1:
                f.write("%d " % (n.numerator))
            else:
                f.write("%d/%d " % (n.numerator, n.denominator))
        f.write("\n")
    f.write("end\n")
    f.close()
