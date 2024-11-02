import numpy as np
from pgd import comb
import pydrake.math
from pydrake.all import AutoDiffXd


def generic_fk_fun(q, analytic_ik, grasp_distance, base_translation):
    tf_goal = analytic_ik.FK(q)
    ang = (180 - 2. * 68.) * np.pi / 180.
    c, s = pydrake.math.cos(ang), pydrake.math.sin(ang)
    tf_goal[:-1,:-1] = tf_goal[:-1,:-1] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) \
                                        @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) \
                                        @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    tf_goal[:-1,-1] = tf_goal[:-1,-1] + tf_goal[:-1,:-1] @ np.array([0, 0, -grasp_distance])
    tf_goal[:-1,-1] = tf_goal[:-1,-1] + base_translation
    return tf_goal

# @jit
def q_to_q_full(q, FK_fun, analytic_ik):
    q_full = np.zeros(14, dtype=type(q[0]))
    q_full[:7] = q[:7]
    
    if len(q) == 8:
        tf_goal = FK_fun(q_full[:7])
    else:
        tf_goal = FK_fun(q_full[:7], q[8])
    GC2 = GC4 = GC6 = -1
    psi = q[7]
    q1 = analytic_ik.IK(tf_goal, [GC2, GC4, GC6], psi)
    
    q_full[7:] = q1
    # print(q_full)
    return q_full


iiwa_alpha = np.array([
	-np.pi/2,
	np.pi/2,
	np.pi/2,
	-np.pi/2,
	-np.pi/2,
	np.pi/2,
	0
])
# NOTE: We're using the LBR iiwa 14 R820, so our values are slightly different
# than the report found here: https://zenodo.org/record/4063575
iiwa_d = np.array([
	0.36,
	0,
	0.42,
	0,
	0.4,
	0,
	0.126-0.045 # This adjustment is necessary to match drake. Probably due to a flange or something?
])
iiwa_limits_lower = np.array([
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-3.054326
])
iiwa_limits_upper = np.array([
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	3.054326
])

def cross_product_matrix(a):
	# Returns a matrix, such that multiplication by a vector yields the cross product
	# See: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
	# Taken from https://stackoverflow.com/questions/66707295/numpy-cross-product-matrix-function#comment117919990_66707295
	return np.cross(a, np.identity(a.shape[0]) * -1)

def scalar_clip(val, a, b):
	if type(val) == AutoDiffXd:
		a = AutoDiffXd(a, np.zeros(val.derivatives().shape))
		b = AutoDiffXd(b, np.zeros(val.derivatives().shape))
	return pydrake.math.max(
		a, pydrake.math.min(
			b, val
		)
	)

class Analytic_IK_JAXLESS():
	# Class that performs analytic forwrad and inverse kinematics for
	# a S-R-S 7-DoF manipulator.
	def __init__(self, alpha, d, limits_lower, limits_upper):
		# alpha and d are the DH parameters. We assume all other parameters are zero
		# limits_lower and limits_upper encode the joint limits
		# All arguments should be length seven numpy arrays
		assert len(alpha) == len(d) == len(limits_lower) == len(limits_upper) == 7
		assert d[1] == d[3] == d[5] == 0
		self.alpha = alpha.copy()
		self.d = d.copy()
		self.d_bs, self.d_se, self.d_ew, self.d_wf = d[0], d[2], d[4], d[6]
		self.limits_lower = limits_lower.copy()
		self.limits_upper = limits_upper.copy()

		self.Ts = [
			lambda ti, ai=ai, di=di : np.array([
				[pydrake.math.cos(ti), -pydrake.math.sin(ti)*pydrake.math.cos(ai), pydrake.math.sin(ti)*pydrake.math.sin(ai), 0],
				[pydrake.math.sin(ti), pydrake.math.cos(ti)*pydrake.math.cos(ai), -pydrake.math.cos(ti)*pydrake.math.sin(ai), 0],
				[0, pydrake.math.sin(ai), pydrake.math.cos(ai), di],
				[0, 0, 0, 1]
			])
			for ai, di in zip(self.alpha, self.d)
		]
	
	def FK(self, thetas):
		# thetas should be the joint angles
		# Returns the transform from the base frame B to the end effector frame E, X_EB
		# Also returns the global configuration parameters and elbow angle psi
		eval_Ts = [eval_T(t) for (eval_T, t) in zip(self.Ts, thetas)]
		full_mat = np.linalg.multi_dot(eval_Ts)
		return full_mat

	def IK(self, rigid_transform, GC, psi, return_sw_mats=False, check_clip=-1):
		# Set check_clip to 1, 2, 3, or 4 to look at the input to the arccos function
		thetas = [0 for _ in range(7)]
		GC2, GC4, GC6 = GC

		p_02 = np.array([0, 0, self.d_bs])
		p_24 = np.array([0, self.d_se, 0])
		p_46 = np.array([0, 0, self.d_ew])
		p_67 = np.array([0, 0, self.d_wf])

		p_07 = rigid_transform[:-1,-1]
		R_07 = rigid_transform[:-1,:-1]
		p_26 = p_07 - p_02 - (R_07 @ p_67) # EQ (3)
		p_26_hat = p_26 / np.linalg.norm(p_26)

		theta_1v = pydrake.math.arctan2(p_26[1], p_26[0]) # EQ (5)

		# EQ (7)
		arccos_in = (self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2) / (2 * self.d_se * np.linalg.norm(p_26))
		if check_clip == 1:
			return arccos_in
		phi = pydrake.math.arccos(scalar_clip(arccos_in, -1, 1))
		theta_2v = pydrake.math.arctan2(np.linalg.norm(p_26[:2]), p_26[2]) + (GC4 * phi)

		theta_3v = 0
		
		# EQ (4)
		arccos_in = (np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
		if check_clip == 2:
			return arccos_in
		theta_4v = GC4 * pydrake.math.arccos(scalar_clip(arccos_in, -1, 1))
		thetas[3] = theta_4v

		theta_vs = [theta_1v, theta_2v, theta_3v, theta_4v]
		T_vs = [T(theta_v) for T, theta_v in zip(self.Ts[:len(theta_vs)], theta_vs)]
		T_03_v = np.linalg.multi_dot(T_vs[0:3])
		R_03_v = T_03_v[:-1,:-1]

		# EQ (15)
		cprod_p_26 = cross_product_matrix(p_26_hat)
		A_s = cprod_p_26 @ R_03_v
		B_s = -1 * cprod_p_26 @ cprod_p_26 @ R_03_v
		C_s = np.outer(p_26_hat, p_26_hat) @ R_03_v

		# EQ (17)-(19)
		thetas[0] = pydrake.math.arctan2(
			GC2 * (A_s[1,1] * pydrake.math.sin(psi) + B_s[1,1] * pydrake.math.cos(psi) + C_s[1,1]),
			GC2 * (A_s[0,1] * pydrake.math.sin(psi) + B_s[0,1] * pydrake.math.cos(psi) + C_s[0,1])
		)
		arccos_in = A_s[2,1] * pydrake.math.sin(psi) + B_s[2,1] * pydrake.math.cos(psi) + C_s[2,1]
		if check_clip == 3:
			return arccos_in
		thetas[1] = GC2 * pydrake.math.arccos(scalar_clip(arccos_in, -1, 1))
		thetas[2] = pydrake.math.arctan2(
			GC2 * (-A_s[2,2] * pydrake.math.sin(psi) - B_s[2,2] * pydrake.math.cos(psi) - C_s[2,2]),
			GC2 * (-A_s[2,0] * pydrake.math.sin(psi) - B_s[2,0] * pydrake.math.cos(psi) - C_s[2,0])
		)

		# EQ (20)
		T_34 = T_vs[3]
		R_34 = T_34[:-1,:-1]
		A_w = R_34.T @ A_s.T @ R_07
		B_w = R_34.T @ B_s.T @ R_07
		C_w = R_34.T @ C_s.T @ R_07

		# EQ (22)-(24)
		thetas[4] = pydrake.math.arctan2(
			GC6 * (A_w[1,2] * pydrake.math.sin(psi) + B_w[1,2] * pydrake.math.cos(psi) + C_w[1,2]),
			GC6 * (A_w[0,2] * pydrake.math.sin(psi) + B_w[0,2] * pydrake.math.cos(psi) + C_w[0,2])
		)
		arccos_in = A_w[2,2] * pydrake.math.sin(psi) + B_w[2,2] * pydrake.math.cos(psi) + C_w[2,2]
		if check_clip == 4:
			return arccos_in
		thetas[5] = GC6 * pydrake.math.arccos(scalar_clip(arccos_in, -1, 1))
		thetas[6] = pydrake.math.arctan2(
			GC6 * (A_w[2,1] * pydrake.math.sin(psi) + B_w[2,1] * pydrake.math.cos(psi) + C_w[2,1]),
			GC6 * (-A_w[2,0] * pydrake.math.sin(psi) - B_w[2,0] * pydrake.math.cos(psi) - C_w[2,0])
		)

		if return_sw_mats:
			return np.asarray(thetas), A_s, B_s, C_s, A_w, B_w, C_w
		else:
			return np.asarray(thetas)


# I could probably make a different file to track these parameters
order = 3
ndim = 8
sampling_resolution = 10

analytic_ik_JAXLESS = Analytic_IK_JAXLESS(iiwa_alpha, iiwa_d, iiwa_limits_lower, iiwa_limits_upper)
FK_fun_JAXLESS = lambda q : generic_fk_fun(q, analytic_ik_JAXLESS, 0.5, np.array([0, -0.765, 0]))

def get_gammas(x, s, s_next):
    x_dim = 8*(order+1)
    x = np.asarray(x)
    # print(type(x[0]))
    assert len(x) == x_dim
    x = x.reshape((-1, 8))
    gamma_s = 0
    gamma_s_next = 0
    for k in range(order + 1):
        gamma_s += comb(order, k) * s**k * (1-s)**(order-k) * x[k]
        gamma_s_next += comb(order, k) * s_next**k * (1-s_next)**(order-k) * x[k]
    return gamma_s, gamma_s_next

def true_distance_cost(x, s, s_next):
    # x are my control points :sob:
    gamma_s, gamma_s_next = get_gammas(x, s, s_next)
    full_s = q_to_q_full(gamma_s.flatten(), FK_fun_JAXLESS, analytic_ik_JAXLESS)
    full_s_next = q_to_q_full(gamma_s_next.flatten(), FK_fun_JAXLESS, analytic_ik_JAXLESS)
    return ((- full_s + full_s_next).dot(- full_s + full_s_next))    

sampling_resolution = 10
# vertex path length
# @jit
def distance_for_vertex(x, sr=sampling_resolution, squared=True):
    # x being the (order + 1) * 8 variables for the ctrl points of given vertex
    cost = 0
    for i in range(sr):
        s = 1.0/sr * i
        s_next = 1.0/sampling_resolution * (i+1)
        cost += true_distance_cost(x, s, s_next) if squared else true_distance_cost(x, s, s_next)**0.5
    return cost

def bezier_derivative(bezier):
    d = bezier.shape[0] - 1
    derivative_controls = []
    for k in range(d):
        bdk = d*(bezier[k+1]-bezier[k])
        derivative_controls.append(bdk)
    return np.asarray(derivative_controls)

from pgd import get_curvature
def max_curvature_for_vertex(x, sr=sampling_resolution):
    curvature = 0
    x_dim = 8*(order+1)
    x = np.asarray(x)
    # print(type(x[0]))
    assert len(x) == x_dim
    x = x.reshape((-1, 8))
    fd = bezier_derivative(x)
    sd = bezier_derivative(fd)

    for i in range(sr):
        s = i/float(sr)
        r = get_curvature(s, fd, sd)
        curvature += np.exp(r)
    
    return np.log(curvature)
