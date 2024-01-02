import tensorflow as tf
import numpy      as np
import trieste

from   pymoo.problems     import get_problem
from   pymoo.core.problem import Problem
from   config             import Setting

class DEB2DK(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    
    bounds = [[0] * Setting.n_var, [1] * Setting.n_var]
    dim    = Setting.n_var

    def objective(self):
         return deb2dk
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return deb2dk(tf.concat([_x, _x], axis=1))
    
def deb2dk(x):
    g  = 1 + (9/(Setting.n_var - 1)) * tf.reduce_sum(x[:, 1:], axis = 1)
    r  = 5 + 10 * ((x[:,0] - 0.5) ** 2) + (1/Setting.knees) * tf.math.cos(2 * Setting.knees * np.pi * x[:,0])
    f1 = g * r * tf.math.sin(np.pi * x[:,0] / 2) 
    f2 = g * r * tf.math.cos(np.pi * x[:,0] / 2)
    return tf.stack([f1, f2], axis=-1)


class DO2DK(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):

    bounds = [[0] * Setting.n_var, [1] * Setting.n_var]
    dim    = Setting.n_var

    def objective(self):
        return do2dk
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return do2dk(tf.concat([_x, _x], axis=1))
    
def do2dk(x):
    g  = 1 + (9/(Setting.n_var - 1)) * np.sum(x[:, 1:], axis = 1)
    r  = 5 + 10 * ((x[:,0] - 0.5) ** 2) + (1/Setting.knees) * np.cos(np.pi * 2 * Setting.knees * x[:,0]) * (2 ** (1/2))
    f1 = g * r * (np.sin((np.pi * x[:,0]) / (2 ** (1+1)) + (1+ (2 ** Setting.s -1) / (2 ** (1+2))) * np.pi) + 1)
    f2 = g * r * (np.cos(np.pi * x[:,0] / 2 + np.pi) + 1)
    return tf.stack([f1, f2], axis=-1)


class DEB3DK(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    
    bounds = [[0] * Setting.n_var, [1] * Setting.n_var]
    dim    = Setting.n_var

    def objective(self):
        return deb3dk
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return deb3dk(tf.concat([_x, _x], axis=1))
    
def deb3dk(x):
    g  = 1 + (9/(Setting.n_var - 1)) * np.sum(x[:, 1:], axis = 1)
    r0 = 5 + 10 * ((x[:,0] - 0.5) ** 2) + (2/Setting.knees) * np.cos(2 * Setting.knees * np.pi * x[:,0])
    r1 = 5 + 10 * ((x[:,1] - 0.5) ** 2) + (2/Setting.knees) * np.cos(2 * Setting.knees * np.pi * x[:,1])
    r  = (r1 + r0)/2
    f1 = g * r * np.sin(np.pi * x[:,0] / 2) * np.sin(np.pi * x[:,1] / 2)
    f2 = g * r * np.cos(np.pi * x[:,1] / 2) * np.sin(np.pi * x[:,0] / 2)
    f3 = g * r * np.cos(np.pi * x[:,0] / 2)
    return tf.stack([f1, f2, f3], axis=-1)


class Cyclone(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    
    bounds = [[0.4, 0.14, 0.4, 3.0, 1.0, 0.4, 0.2], [0.7, 0.4, 0.75, 7.0, 2.0, 2.0, 0.4]]
    dim    = Setting.n_var

    def objective(self):
        return cyclone
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return cyclone(tf.concat([_x, _x], axis=1))
    
def cyclone(x):
    D = 31e-3

    a  = x[:,0] * D
    b  = x[:,1] * D
    Dx = x[:,2] * D
    Ht = x[:,3] * D
    h  = x[:,4] * D
    S  = x[:,5] * D
    Bc = x[:,6] * D

    flow = 0.83333333
    Vin  = flow * 1e-3 / (a * b)

    ro_g = 1.225
    ro_p = 860
    mu_g = 1.7894e-5
    c0   = 1200
    g    = 9.81

    volume_flow_rate = Vin * a * b

    R   = D / 2.0
    Rx  = Dx / 2.0
    Rin = R - 0.5 * b
    RB  = Bc / 2.0
    c0  = c0 / 1000.0

    xmed = 6.0e-6
    ks   = 0.1e-3

    z    = b / R
    x1   = (1.0 - z * z) * (2.0 * z - z * z) / (1.0 + c0)
    x2   = z * z / 4.0 - z / 2.0
    alfa = (1.0 / z) * (1.0 - (1.0 + 4.0 * x2 * (1.0 - x1) ** (0.5)) ** (0.5))


    Vx = volume_flow_rate / (np.pi * Rx * Rx)
    Rm = (Rx * R) ** (0.5)
    Y  = (R - Rx) * (Ht - h) / (R - (Bc / 2.0))

    mask = Bc > Dx
    Y    = tf.where(mask, Ht - h, Y)
    Hcs  = Y + (h - S)


    Vtw = (Vin * Rin) / (alfa * R)
    Vzw = 0.9 * volume_flow_rate / (np.pi * (R * R - Rm * Rm))

    Vtcs = Vin / 4
    Vtm  = (Vtw * Vtcs) ** (0.5)
    x1   = 1 + (Vzw / Vtm) ** 2
    ReR  = (Rin * Rm * Vzw * ro_g) / (Ht * mu_g * x1)


    ksR = ks / R
    if ksR < 6e-4:
        ksR = 6e-4

    fsm = 0.323 * (ReR) ** (-0.623)
    fr  = (2.38 * tf.experimental.numpy.log10(1.6 / (ksR - 0.000599))) ** (-2)
    x3  = ReR * ReR * (ksR - 0.000599) ** (0.213)
    fr  = fr * 1.0 / (1.0 + 2.25e5 / x3)

    ro_bulk = 0.5 * ro_p
    ro_str  = 0.4 * ro_bulk
    Frx     = Vx / (2.0 * Rx * g)
    eta     = 0.99
    fair    = fsm + fr
    f       = fair + 0.25 * (R / Rx) ** (-0.625) * (eta * c0 * Frx * ro_g / ro_str) ** (0.5)

    x1 = R * R - Rx * Rx
    x2 = 2.0 * R * h
    x3 = (R + RB) * ((Ht - h) ** (2) + (R - RB) ** (2)) ** (0.5)
    x4 = 2.0 * Rx * S
    AR = np.pi * (x1 + x2 + x3 + x4)
    x4 = f * AR * Vtw * (R / Rx) ** (0.5) / (2.0 * volume_flow_rate)

    Vtcs  = Vtw * (R / Rx) / (1.0 + x4)
    xfact = 1.0

    x1 = 18.0 * mu_g * 0.9 * volume_flow_rate
    x2 = 2.0 * np.pi * (ro_p - ro_g) * Vtcs * Vtcs * (Ht-S)

    MM_x50 = xfact * (x1 / x2) ** (0.5)
    Ut50   = volume_flow_rate / (2.0 * np.pi * Rx * Hcs)
    Rep    = ro_g * Ut50 * MM_x50 / mu_g
    
    mask   = Rep > 0.5
    x1     = tf.where(mask, 5.18 * (mu_g) ** (0.375) * (ro_g) ** (0.25) * (Ut50) ** (0.875), x1)
    x2     = tf.where(mask, ((ro_p - ro_g) * Vtcs * Vtcs) / Rx, x2)
    MM_x50 = tf.where(mask, x1/x2, MM_x50)

    if (c0 < 0.1):
        k   = -0.11 - 0.1 * tf.math.log(c0)
        coL = 0.025 * (MM_x50 / xmed) * (10.0 * c0) ** (k)
    else:
        coL = 0.025 * (MM_x50 / xmed) * (10.0 * c0) ** (0.15)

    x1 = f * AR * ro_g * (Vtw * Vtcs) ** (1.5)
    x2 = 2.0 * 0.9 * volume_flow_rate
    dp_body = x1 / x2

    x1    = Vtcs / Vx
    x2    = 0.5 * ro_g * Vx * Vx
    dp_x  = (2.0 + (x1 * x1) + 3.0 * (x1) ** ((4. / 3.))) * x2
    MM_dp = dp_body + dp_x

    Eu = MM_dp / (0.5 * ro_g * Vin * Vin)
    f1 = tf.math.log(Eu)

    vt_max = 6.1 * Vin * (((a * b / (D * D)) ** (0.61)) * ((Dx / D) ** (-0.74)) * ((Ht / D) ** (-0.33)))

    dc = 0.47 * D * ((a * b / (D *D)) ** (-0.26)) * ((Dx / D) ** 1.4)
    zc = Ht - S
    
    mask = dc > Bc
    zc   = tf.where(mask, (Ht - S) - ((Ht - h) / ((D / Bc) - 1.0)) * ((Dx / Bc) - 1.0), zc) 

    X50_Iozia        = ((9.0 * mu_g * volume_flow_rate) / (np.pi * zc * ro_p * vt_max * vt_max)) ** 0.5
    Stk50_Iozia_1000 = 1000 * ro_p * X50_Iozia ** 2 * Vin / (18 * mu_g * D)

    f2 = tf.math.log(Stk50_Iozia_1000)
    
    return tf.stack([f1, f2], axis=-1)


class Cbeam(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    
    bounds = [[0] * 3, [1] * 3]
    dim    = 3

    def objective(self):
        return cbeam
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return cbeam(tf.concat([_x, _x], axis=1))
    
def cbeam(x):
    h1 = 1.0
    L  = 36
    E  = 1.0e7
    P  = 1000.0
    H  = x[:,0] * 4 + 3
    b1 = x[:,1] * 10 + 2
    b2 = x[:,2] * 1.9 + 0.1

    Volume    = (2 * h1 * b1 + (H - 2 * h1) * b2) * L
    I         = 1.0 / 12.0 * b2 * (H - 2 * h1)**3 + 2 * (1.0 / 12.0 * b1 * h1**3 + b1 * h1 * (H - h1)**2 / 4)
    MaxStress = P * L * H / (2 * I)
    return tf.stack([Volume, MaxStress/10], axis=-1)

class DTLZ7(trieste.objectives.multi_objectives.MultiObjectiveTestProblem):
    
    bounds = [[0] * Setting.n_var, [1] * Setting.n_var]
    dim    = Setting.n_var

    def objective(self):
        return dtlz7
    
    def gen_pareto_optimal_points(self, n, seed = None):
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return dtlz7(tf.concat([_x, _x], axis=1))
    
def dtlz7(x):
    if type(x) != type(np.array([])):
        return tf.convert_to_tensor(get_problem("dtlz7", n_var = Setting.n_var, n_obj = 2).evaluate(x.numpy()))
    return tf.convert_to_tensor(get_problem("dtlz7", n_var = Setting.n_var, n_obj = 2).evaluate(x))

class CKPNSGAII(Problem):
    def __init__(self, n_var, knees):
        super().__init__(n_var = n_var, n_obj = 2, n_constr = 0, xl = np.array([0] * n_var), xu = np.array([1] * n_var))
        self.knees = knees
        
    def _evaluate(self, x, out, *args, **kwargs):
        g  = 1 + (9/(self.n_var - 1)) * np.sum(x[:, 1:], axis = 1)
        r  = 5 + (x[:,0] ** 2) + (1/self.knees) * np.cos(2 * self.knees * np.pi * x[:,0])
        f1 = g * r * np.sin(np.pi * x[:,0] / 2) 
        f2 = g * r * np.cos(np.pi * x[:,0] / 2)
        
        out["F"] = np.column_stack([f1, f2])

def ckp(x):
    if type(x) != type(np.array([])):
        return tf.convert_to_tensor(CKPNSGAII(n_var = Setting.n_var, knees = Setting.knees).evaluate(x.numpy()))
    return tf.convert_to_tensor(CKPNSGAII(n_var = Setting.n_var, knees = Setting.knees).evaluate(x))

def zdt3(x):
    if type(x) != type(np.array([])):
        return tf.convert_to_tensor(get_problem("zdt3", n_var = Setting.n_var).evaluate(x.numpy()))
    return tf.convert_to_tensor(get_problem("zdt3", n_var = Setting.n_var).evaluate(x))

