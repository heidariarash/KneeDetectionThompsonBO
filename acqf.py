import trieste
import tensorflow as tf
import numpy      as np

from   config                     import Setting
from   pymoo.algorithms.moo.nsga2 import NSGA2
from   pymoo.algorithms.moo.sms   import SMSEMOA
from   pymoo.optimize             import minimize
from   pymoo.termination          import get_termination
from   pymoo.core.problem         import Problem


class HvKnee(trieste.acquisition.SingleModelAcquisitionBuilder):
    def __repr__(self) -> str:
        """"""
        return "HV-Knee()"

    def prepare_acquisition_function(self, model, dataset):
        _pf               = trieste.acquisition.multi_objective.non_dominated(dataset.observations)[0].numpy()
        _reference_pt     = tf.reduce_max(_pf, axis = 0, keepdims= False)
        _partition_bounds = trieste.acquisition.multi_objective.pareto.prepare_default_non_dominated_partition_bounds(_reference_pt, tf.reshape(_reference_pt, (1, -1)))
        return trieste.acquisition.function.expected_hv_improvement(model, _partition_bounds)


class ExtremaFinder(trieste.acquisition.SingleModelAcquisitionBuilder):
    def __repr__(self) -> str:
        """"""
        return "ExtremaFinder()"

    def prepare_acquisition_function(self, model, dataset):
        _reference_pt     = tf.constant([10000, 10000], dtype = tf.float64)
        _pf               = tf.reduce_min(dataset.observations, 0, keepdims = True)
        _partition_bounds = trieste.acquisition.multi_objective.pareto.prepare_default_non_dominated_partition_bounds(_reference_pt, _pf)
        return trieste.acquisition.function.expected_hv_improvement(model, _partition_bounds)
    
class HVThompson(trieste.acquisition.SingleModelAcquisitionBuilder):
    """
    Hyper Volume Thompson Sampling
    """

    def __repr__(self) -> str:
        """"""
        return "Hyper Volume Thompson Sampling()"

    def prepare_acquisition_function(self, model, dataset = None):
        mins         = [0] * Setting.n_var
        maxs         = [1] * Setting.n_var
        if Setting.problem == "cyclone":
            mins = [0.4, 0.14, 0.4, 3.0, 1.0, 0.4, 0.2]
            maxs = [0.7, 0.4, 0.75, 7.0, 2.0, 2.0, 0.4]
        search_space = trieste.space.Box(mins, maxs)
        
        samples = search_space.sample_halton(50000)
        ts0     = model._models[0].trajectory_sampler().get_trajectory()
        ts1     = model._models[1].trajectory_sampler().get_trajectory()
        if Setting.problem == "DEB3DK":
            ts2 = model._models[2].trajectory_sampler().get_trajectory()
        eval0   = ts0(samples)
        eval1   = ts1(samples)
        if Setting.problem == "DEB3DK":
            eval2 = ts2(samples)
        max1    = eval1[tf.argmin(eval0)]
        max0    = eval0[tf.argmin(eval1)]
        
        if Setting.problem == "DEB3DK":
            max0 = max(eval0[tf.argmin(eval1)].numpy(), eval0[tf.argmin(eval2)].numpy())
            max1 = max(eval1[tf.argmin(eval0)].numpy(), eval1[tf.argmin(eval2)].numpy())
            max2 = max(eval2[tf.argmin(eval0)].numpy(), eval2[tf.argmin(eval1)].numpy())

        _reference_pt     = tf.constant([max0.numpy(), max1.numpy()], dtype = tf.float64)
        if Setting.problem == "DEB3DK":
            _reference_pt     = tf.constant([max0, max1, max2], dtype = tf.float64)
        _partition_bounds = trieste.acquisition.multi_objective.pareto.prepare_default_non_dominated_partition_bounds(_reference_pt, tf.reshape(_reference_pt, (1, -1)))
        
        return trieste.acquisition.function.expected_hv_improvement(model, _partition_bounds)
    
class HVNSGARule(trieste.acquisition.rule.AcquisitionRule):
    def __init__(self):
        pass

    def __repr__(self) -> str:
        """"""
        return f"""Angle Thompson Rule"""

    def acquire(self, search_space, model, datasets):
        model = model['OBJECTIVE']
        ts0   = model._models[0].trajectory_sampler().get_trajectory()
        ts1   = model._models[1].trajectory_sampler().get_trajectory()
        ts2   = None
        if Setting.problem == "DEB3DK":
            ts2 = model._models[2].trajectory_sampler().get_trajectory()

        problem = self.samplers(ts0, ts1, ts2)
        
        if Setting.EA_util == "NSGAII":
            algorithm   = NSGA2(pop_size = 100)
        else:
            algorithm   = SMSEMOA(pop_size = 100)

        termination = get_termination("n_gen", 200)
        res         = minimize(problem, algorithm, termination, seed = 1, verbose = False)
        
        max0        = tf.reduce_max(res.F[:,0])
        max1        = tf.reduce_max(res.F[:,1])
        if Setting.problem == "DEB3DK":
            max2    = tf.reduce_max(res.F[:,2])
        if Setting.problem == "DEB3DK":
            hvs     = (max0 - res.F[:,0]) * (max1 - res.F[:,1]) * (max2 - res.F[:,2])
        else:
            hvs     = (max0 - res.F[:,0]) * (max1 - res.F[:,1])
        candidate   = res.X[tf.argmax(hvs)].reshape(1,-1)
        
        del res, algorithm, ts0, ts1
        
        return candidate

    def samplers(self, ts0, ts1, ts2 = None):
        
        class EA_Problem(Problem):
            def __init__(self):
                if Setting.problem == "DEB3DK":
                    super().__init__(n_var = Setting.n_var, n_obj = 3, n_constr = 0, xl = np.array([0] * Setting.n_var), xu = np.array([1] * Setting.n_var))
                else:
                    super().__init__(n_var = Setting.n_var, n_obj = 2, n_constr = 0, xl = np.array([0] * Setting.n_var), xu = np.array([1] * Setting.n_var))

            def _evaluate(self, x, out, *args, **kwargs):
                f1 = ts0(x)
                f2 = ts1(x)
                if Setting.problem == "DEB3DK":
                    f3 = ts2(x)
                
                if Setting.problem == "DEB3DK":
                    out["F"] = np.column_stack([f1, f2, f3])
                else:
                    out["F"] = np.column_stack([f1, f2])
        return EA_Problem()