import numpy as np
import time
# import pickle
# import os
# from numba import jit
from pydrake.common import RandomGenerator

import jax.numpy as jnp
import jax
from time import perf_counter

from pydrake.all import (
    MathematicalProgram,
    Solve
)

from pydrake.trajectories import BezierCurve, CompositeTrajectory

from pydrake.all import GcsTrajectoryOptimization, GraphOfConvexSetsOptions, Point, GraphOfConvexSets

# from src.common import (
#     generic_fk_fun,
#     q_to_q_full
# )
# from src.analytic_ik import Analytic_IK_7DoF, iiwa_alpha, iiwa_d, iiwa_limits_lower, iiwa_limits_upper
# from src.pgd import *
import random as rand

order = 3
# FK_fun = lambda q : generic_fk_fun(q, analytic_ik, 0.5, np.array([0, -0.765, 0]))
# analytic_ik = Analytic_IK_7DoF(iiwa_alpha, iiwa_d, iiwa_limits_lower, iiwa_limits_upper)
ndim = 8

def comb(n, r):
    # n factorial / r factorial * n - r factorial
    # same as n * ... * n - r + 1 / r * .... * 1
    # we assume neither n nor r are negative ！ as they shouldn't be 
    result = 1.0
    for m in range(n - r + 1, n+1):
        result *= m
    for q in range(1, r+1):
        result = result / q
    return result

def generate_flows(start, goal, order, regions, dim=8):
    # t1 = time.time()
    continuity = 1

    gcs = GcsTrajectoryOptimization(dim)
    if continuity > 0:
        gcs.AddPathContinuityConstraints(continuity)
    if dim==8:
        try:
            wraparound = np.full(8, np.inf)
            wraparound[-1] = 2*np.pi
            main_graph = gcs.AddRegions(regions, order, h_min=0.1, h_max=100, name="", wraparound=wraparound)
        except:
            main_graph = gcs.AddRegions(regions, order, h_min=0.1, h_max=100, name="")
    else:
        main_graph = gcs.AddRegions(regions, order, h_min=0.1, h_max=100, name="")
    start_graph = gcs.AddRegions([Point(start)], 0)
    goal_graph = gcs.AddRegions([Point(goal)], 0)
    gcs.AddEdges(start_graph, main_graph)
    gcs.AddEdges(main_graph, goal_graph)

    gcs.AddPathLengthCost()
    if dim==8:
        gcs.AddVelocityBounds(-np.ones(dim), np.ones(dim))

#     s = gcs.GetGraphvizString()
#     plot_dot(s)

    options = GraphOfConvexSetsOptions()
    options.max_rounding_trials = 1000
    options.max_rounded_paths = 100
    options.convex_relaxation = True

    _, result = gcs.SolvePath(start_graph, goal_graph, options)

#     s = gcs.GetGraphvizString(result, show_slack=False)
#     plot_dot(s)

    # t2 = time.time()
    
    return gcs, result

def prune(gcs, flows_result, epsilon=0.00001):
    # first we prune the edges and vertices
    for e in gcs.graph_of_convex_sets().Edges():
        if flows_result.GetSolution(e.phi()) < epsilon:
            gcs.graph_of_convex_sets().RemoveEdge(e)
    for v in gcs.graph_of_convex_sets().Vertices():
        if (len(v.incoming_edges()) + len(v.outgoing_edges())) <= 0:
            gcs.graph_of_convex_sets().RemoveVertex(v)
    # find the start and end vertices
    for v in gcs.graph_of_convex_sets().Vertices():
        # these names should still hold as there is one region in both start and goal regions
        # and they are added 1st and 2nd respectively
        if v.name() == "Subgraph1: Region0":
            start_vertex = v 
        if v.name() == "Subgraph2: Region0":
            end_vertex = v
    print("Start vertex is ", start_vertex, "  and end vertex is ", end_vertex)
    return start_vertex, end_vertex

def run_dfs(flows_result, gcs, start_vertex, end_vertex, iterations=10):
    # Rounding step:  run DFS on the flows generated
    # To get the vertices path and initial result to initialise SNOPT
    # ATM: using the SolveConvexRelaxation to evaluate optimality
    paths_found = []
    best_rounded_result = None
    discarded_edges = set()
    best_path = []
    for i in range(iterations):
    #     print(f"ITERtATION {i}")
        current_vertex = start_vertex
        current_path = []
        current_path_vertices = [start_vertex]
        visited_nodes = set([start_vertex])
        # stop when we're at goal region currently called so.
        while current_vertex.name() != "Subgraph2: Region0":

            edges = []
            for e in current_vertex.outgoing_edges():
                if e.v() not in visited_nodes:
                    edges.append(e)
    #         print(current_vertex.name(), [e.v().name() for e in edges])

            if len(edges) == 0:
                # if no path forward, we backtrack
                # go back to the last 
                current_path_vertices.pop() 
                visited_nodes.add(current_vertex)
                current_vertex = current_path_vertices[-1]
                # take out the last edge and don't visit it again
                discarded_edges.add(current_path.pop())
            elif len(edges) == 1:
                # if there's only path forward we take that
                current_path.append(edges[0])
                visited_nodes.add(current_vertex)
                current_vertex = edges[0].v()
                current_path_vertices.append(current_vertex)
            else:
                # we've multiple paths and need to choose
                e_c = [flows_result.GetSolution(e.phi()) for e in edges]
                edge_sample = rand.uniform(0, 1) * sum(e_c)
                for i in range(len(e_c)):
                    if e_c[i] <= edge_sample:
                        edge_sample -= e_c[i]
                    else:
                        current_path.append(edges[i])
                        visited_nodes.add(current_vertex)
                        current_vertex = edges[i].v()
                        current_path_vertices.append(current_vertex)

        if len(best_path) == 0:
            best_path = current_path_vertices

        rounded_result = GraphOfConvexSets.SolveConvexRestriction(gcs.graph_of_convex_sets(), current_path, GraphOfConvexSetsOptions())

        if rounded_result.is_success() and ((best_rounded_result is None) or best_rounded_result.get_optimal_cost() > rounded_result.get_optimal_cost()):
            best_rounded_result = rounded_result
            best_path = current_path_vertices
    #         print("BETTER PATH FOUND : ", [c.name() for c in best_path])

    print("Cost for best path found  ", best_rounded_result.get_optimal_cost())
#     for v in best_path:
#         print(v.name())
        
    return best_path, best_rounded_result

def trajectorify(path, result, ndim=8):
    bezier_curves = []
    i = 0
    for vertex in path[1:-1]:
        
        control_pts = result.GetSolution(vertex.x())[:-1].reshape((-1, ndim)).T
        bezier_curves.append(BezierCurve(i, i+1, control_pts))
        i+=1
    return CompositeTrajectory(bezier_curves)

def prog_with_constraints(variables, best_path, start, goal, indices, ndim = 8, h_list=None):
    v = np.asarray(variables)
    qp = MathematicalProgram()
    sampling_resolution = 10
    eps = 1e-6
    # constraints for the waypoints
    prev = best_path[0]
    qp.AddDecisionVariables(v)
    qp.AddLinearEqualityConstraint(v[:ndim], start)
    qp.AddLinearEqualityConstraint(v[len(v)-ndim:], goal)
    if h_list is None:
        h_vars = []

    # # vertex number
    j = 0 
    # # exclude the start and end pt
    for i in range(len(best_path[1:-1])):
        v_i = best_path[i+1]
        if h_list is None:
            h = qp.NewContinuousVariables(1, f'h{i}')
            h_vars.append(h)
        else:
            h = [h_list[i]]
            qp.AddDecisionVariables(h)
        
        temp = [elem for elem in v[indices[i][0]:indices[i][1]]]
        temp.extend(h)
        v_i.set().AddPointInSetConstraints(qp, temp)

        # we don't want to put a velocity constraint from the first vertex which is just the start point
    #     the end point constraint is added outside the loop so we don't need to skip it here
    # DOUBLE CHECK THE INDICES HERE
        pivot = indices[i][1]
        # if j!=0 and j!=len(best_path[1:-1])-1:
        #     x_dot_d = order * (np.asarray(v[pivot-ndim:pivot]) - v[pivot-ndim*2:pivot-ndim])
        #     x_dot_0 = order * (np.asarray(v[pivot:pivot+ndim]) - v[pivot-ndim:pivot] )
        #     qp.AddLinearEqualityConstraint(x_dot_0-x_dot_d, np.zeros((ndim, 1)))

        j+=1
    if h_list is None:
        h_list = h_vars

    return qp, h_list
    
def run_qp_proj(variables, init_values, best_path, start, goal, indices, h_vars=None, prog=None, cost=None):
    v = np.asarray(variables)
    if prog is None:
        qp, _ = prog_with_constraints(variables, best_path, start, goal, indices, h_list=h_vars)
    else:
        qp = prog
        if len(qp.quadratic_costs())>0:
            if cost is None:
                qp.RemoveCost(qp.quadratic_costs()[0])
            else:
                qp.RemoveCost(cost)
            # print("I am doing what I am supposed to")

    cost = qp.AddQuadraticErrorCost(np.identity(len(v)), init_values, v)
    # print(type(c))
    # if h_vars is None:
    #     qp.SetInitialGuess(v, init_values)
    # else:
    #     qp.SetInitialGuess(v, init_values[:len(variables)])
    #     qp.SetInitialGuess(np.asarray(h_vars), init_values[len(variables):])
    projection = Solve(qp)
    if h_vars is None:
        return [projection.GetSolution(x) for x in variables]
    res = [projection.GetSolution(x) for x in variables]
    res.extend([projection.GetSolution(h) for h in h_vars])
    return res, cost


# def project_qp()

# @jit() imprecise type as we have two types of parameters being passed in
# np.float64[:], and pydrake.autodiffutils.AutoDiffXd[:]
# can maybe try overloading
def get_gammas(x, s, s_next, ndim=2):
    x_dim = ndim*(order+1)
    x = jnp.asarray(x)
    # print(type(x[0]))
    # print(len(x), x_dim)
    assert len(x) == x_dim
    x = x.reshape((-1, ndim))
    gamma_s = 0
    gamma_s_next = 0
    for k in range(order + 1):
        gamma_s += comb(order, k) * s**k * (1-s)**(order-k) * x[k]
        gamma_s_next += comb(order, k) * s_next**k * (1-s_next)**(order-k) * x[k]
    return gamma_s, gamma_s_next

# If we're planning through t
# we want to go from t to theta
# 2 * invtan t
q_star = np.array([0., 0., 0., 0., 0., 0., 0.])
def true_distance_cost(x, s, s_next, ndim=7):
    # x are my control points :sob:
    gamma_s, gamma_s_next = get_gammas(x, s, s_next, ndim=ndim)
    full_s = 2*jnp.arctan2(gamma_s.flatten() , jnp.ones((len(gamma_s),))) + q_star
    full_s_next = 2*jnp.arctan2(gamma_s_next.flatten(), jnp.ones((len(gamma_s),))) + q_star
    # jax.debug.print("{x}", x=((- full_s + full_s_next).dot(- full_s + full_s_next)))
    # jax.debug.print("-------")
    # print(((- full_s + full_s_next).dot(- full_s + full_s_next))**0.5)
    return ((- full_s + full_s_next).dot(- full_s + full_s_next))**0.5

sampling_resolution = 10
# vertex path length
# @jit
def distance_for_vertex(x, sr=sampling_resolution, squared=True):
    # x being the (order + 1) * 8 variables for the ctrl points of given vertex
    cost = 0
    for i in range(sr):
        s = 1.0/sr * i
        s_next = 1.0/sr * (i+1)
        cost += true_distance_cost(x, s, s_next, ndim=7) if squared else true_distance_cost(x, s, s_next)**0.5
    return cost

def get_step(values, indices, best_path, cost_func, gradient_func, step_size=0.01, backtracking=False):
    # R is a delay operator
    df_Rk = []
    last_common = 0
    f_Rk = 0
    # values = np.asarray(values)
    # gradient_funcs = [grad(distance_for_vertex)]
    # t1 = time.perf_counter()
    for v in range(len(best_path[1:-1])):
        # print("v", end='')
        # gradients = fd.gradient(distance_for_vertex, values[indices[v][0]:indices[v][1]])
        gradients = gradient_func(values[indices[v][0]:indices[v][1]])
        # print(len(gradients), len(values[indices[v][0]:indices[v][1]]))
        f_Rk += cost_func(values[indices[v][0]:indices[v][1]])
        if v==0:
            print(gradients)
            df_Rk.extend(gradients)
            # print("should be 32 ", len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
        else:
            # print("adding ", len(gradients[:ndim]), "to ", len(df_Rk[last_common-ndim:]), len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
            # df_Rk[last_common-ndim:] += gradients[:ndim]
            for i in range(ndim):
                df_Rk[last_common-ndim+i] += gradients[i]
            df_Rk.extend(gradients[ndim:])
            # print("should be 24*(v-1) +32 ", len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
        last_common = len(df_Rk)
    # t2 = time.perf_counter()
    # print(t2-t1)
    df_Rk = np.asarray(df_Rk)

    beta = 0.5
    # https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
    if backtracking:
        # print('b', end='')
        t = 0.1
        t1 = time.perf_counter()
        x_k = values - t * df_Rk
        f_k = 0 # TODO: FIX THIS
        for v in range(len(best_path[1:-1])):
            f_k += cost_func(x_k[indices[v][0]:indices[v][1]])
        L_norm = df_Rk.dot(df_Rk)
        while f_k > (f_Rk - 0.5*t*(L_norm)):
            t = beta * t
            x_k = values - t * df_Rk
            f_k = 0
            for v in range(len(best_path[1:-1])):
                f_k += cost_func(x_k[indices[v][0]:indices[v][1]])
            # print(f_k, f_Rk, df_Rk.dot(df_Rk))
        t2 = time.perf_counter()
        # print(t2-t1)

        return x_k, f_k, t2-t1, t, L_norm
    else:
        return df_Rk*step_size*-1

# def get_step_mult_costs(values, indices, best_path, cost_funcs, gradient_funcs, weights, step_size=0.01, backtracking=False):
#     # R is a delay operator
#     df_Rk = []
#     last_common = 0
#     f_Rk = 0
#     # values = np.asarray(values)
#     # gradient_funcs = [grad(distance_for_vertex)]
#     # t1 = time.perf_counter()
#     for v in range(len(best_path[1:-1])):
#         # print("v", end='')
#         # gradients = fd.gradient(distance_for_vertex, values[indices[v][0]:indices[v][1]])
#         gradients = np.asarray(gradient_funcs[0](values[indices[v][0]:indices[v][1]]))
#         # print(type(gradient_funcs[0](values[indices[v][0]:indices[v][1]])))
#         if len(gradient_funcs) > 1:
#             gradients = weights[0]*gradients
#             for i in range(len(gradient_funcs[1:])):
#                 gradients += weights[i+1]*np.asarray(gradient_funcs[i+1](values[indices[v][0]:indices[v][1]]))
#             # print(len(gradients), len(values[indices[v][0]:indices[v][1]]))
#         for i in range(len(cost_funcs)):
#             f_Rk += weights[i]*cost_funcs[i](values[indices[v][0]:indices[v][1]])
#         if v==0:
#             df_Rk.extend(gradients)
#             # print("should be 32 ", len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
#         else:
#             # print("adding ", len(gradients[:ndim]), "to ", len(df_Rk[last_common-ndim:]), len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
#             # df_Rk[last_common-ndim:] += gradients[:ndim]
#             for i in range(ndim):
#                 df_Rk[last_common-ndim+i] += gradients[i]
#             df_Rk.extend(gradients[ndim:])
#             # print("should be 24*(v-1) +32 ", len(df_Rk))#, len(values[indices[v][0]:indices[v][1]]))
#         last_common = len(df_Rk)
#     # t2 = time.perf_counter()
#     # print(t2-t1)
#     df_Rk = np.asarray(df_Rk)

#     beta = 0.5
#     # https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
#     if backtracking:
#         # print('b', end='')
#         t = 0.1
#         t1 = time.perf_counter()
#         x_k = values - t * df_Rk
#         f_k = 0 # TODO: FIX THIS
#         for v in range(len(best_path[1:-1])):
#             for i in range(len(cost_funcs)):
#                 f_k += weights[i]*cost_funcs[i](x_k[indices[v][0]:indices[v][1]])
#         L_norm = df_Rk.dot(df_Rk)
#         while f_k > (f_Rk - 0.5*t*(L_norm)):
#             t = beta * t
#             x_k = values - t * df_Rk
#             f_k = 0
#             for v in range(len(best_path[1:-1])):
#                 for i in range(len(cost_funcs)):
#                     f_k += weights[i]*cost_funcs[i](x_k[indices[v][0]:indices[v][1]])
#             # print(f_k, f_Rk, df_Rk.dot(df_Rk))
#         t2 = time.perf_counter()
#         # print(t2-t1)

#         return x_k, f_k, t2-t1, t, L_norm
#     else:
#         return df_Rk*step_size*-1

# Such an uninformative name
# given a list of values for control points instead of prog result
def trajectorify_given_vars(path, init_vals, indices, ndim=8):
    bezier_curves = []
    i = 0
    for v in range(len(path[1:-1])):
        control_pts = np.asarray(init_vals[indices[v][0]:indices[v][1]]).reshape((-1, ndim)).T
#         print(control_pts)
        bezier_curves.append(BezierCurve(i, i+1, control_pts))
        i+=1
    return CompositeTrajectory(bezier_curves)

def generate_variable_list(best_path, ndim=8):
    # ndim = dim
    variables = [x for x in best_path[1].x()[:ndim]]
    # print(len(best_path[1].x()[:8]))    
    indices = []
    h_list = []
    # list of functions with inputs being the variables
    for i in range(len(best_path[1:-1])):
        start_index = i*(order)*ndim # i = 1, si= 24, i = 2, si = 48
        end_index = ((i+1) *order+1) * ndim # i = 0, ei = 32 ; i = 1, ei = (2 * 3 + 1) * 8 = 56
        indices.append((start_index, end_index))
        v_i = best_path[i+1]
        # print(len(v_i.x()[8:-1]))
        h_list.append(v_i.x()[-1])
        variables.extend(v_i.x()[ndim:-1])# the last variable is the time scaling
#     print()
    assert len(variables) == ndim * (1 + order * len(best_path[1:-1]))
    return variables, indices, h_list

def get_equality_matrices(prog):
    row_size = prog.num_vars()
    equalities = prog.linear_equality_constraints()
    A_eq = []#np.zeros((len(equalities), row_size))
    b_eq = []
    # print(row_size)
#     print()

    for equality in equalities:
        indices = prog.FindDecisionVariableIndices(equality.variables())
        A = equality.evaluator().GetDenseA()
        B = equality.evaluator().upper_bound()
        # print(B)
        for (a, b) in zip(A, B):
            row = np.zeros((1, row_size))
            row[0][indices] += a
            A_eq.append(row[0])#, axis=0)
#             print(len(A_eq), len() len(row[0][indices]), len(a))
            b_eq.append(b)
    return A_eq, np.asarray(b_eq)

# def remap(q):
#   return q_to_q_full(q, FK_fun, analytic_ik)

def generate_samples(regions, number_samples=100):
    n = int(number_samples/len(regions))
    sample_pts = []
    for region in regions:
        samples = []
        generator = RandomGenerator()
        sample = region.ChebyshevCenter()
        for i in range(6):
            sample = region.UniformSample(generator, sample)
            samples.append(sample)
        sample_pts.append(samples)
    print(len(sample_pts))
    return np.asarray(sample_pts)

# def warm_start(fn, init_values):
#     jitfn = jax.jit(fn)
#     t1 = perf_counter()
#     jitfn(init_values)
#     t2 = perf_counter()
#     print("warm start costfn took ", t2 - t1)
#     return jitfn

def bezier_derivative(bezier):
    d = bezier.shape[0] - 1
    derivative_controls = []
    for k in range(d):
        bdk = d*(bezier[k+1]-bezier[k])
        derivative_controls.append(bdk)
    return jnp.asarray(derivative_controls)

def calc_bezier_curve(bezier, s):
    d = len(bezier) - 1
    gamma_s = 0
    # k from 0 to d
    for k in range(d+1):
        gamma_s += comb(d, k) * s**k * (1-s)**(d-k) * bezier[k]
    return gamma_s

# 1/radius of curvature
def get_curvature(s, fd, sd):
    fd_s = calc_bezier_curve(fd, s)
    sd_s = calc_bezier_curve(sd, s)
    fd_norm_sq = fd_s.dot(fd_s)
    sd_norm_sq = sd_s.dot(sd_s)
    denominator = (fd_norm_sq*sd_norm_sq) - (fd_s.dot(sd_s))**2 + 0.00001    
    return (denominator)**0.5 / ((fd_norm_sq)**1.5 + 0.00001)

# two ways to do this perhaps?
# first lets just get the max per vertex and push those up
# second lets get the max per vertex, threshold and push that up.
def max_curvature_for_vertex(x, sr=sampling_resolution):
    curvature = 0
    x_dim = 8*(order+1)
    x = jnp.asarray(x)
    # print(type(x[0]))
    assert len(x) == x_dim
    x = x.reshape((-1, 8))
    fd = bezier_derivative(x)
    sd = bezier_derivative(fd)

    for i in range(sr):
        s = i/float(sr)
        r = get_curvature(s, fd, sd)
        curvature += jnp.exp(r)
    
    return jnp.log(curvature)

# apparently the derivative of the max log exp is softmax. If the gradient gives us grief maybe we can use tht?