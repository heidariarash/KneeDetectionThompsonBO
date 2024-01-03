from acqf     import ExtremaFinder, HvKnee, HVThompson, HVNSGARule
from utils    import calculate_regret_hv, calculate_regret_hv_multiple, calculate_regret_d2l, build_stacked_independent_objectives_model
from problems import DEB2DK, DO2DK, DEB3DK, Cbeam, Cyclone, DTLZ7, ckp, zdt3
from config   import Setting


import trieste
import numpy              as np

def solver(problem: str, method: str, **kwargs):
    ref2 = None

    if problem == "DO2DK":
        num_obj   = 2
        problem   = DO2DK().objective()
        observer  = trieste.objectives.utils.mk_observer(problem)
        hv_best   = np.array([1.06634321, 1.02033657])
        dl_best   = np.array([1.06634321, 1.02033657])
        slope     = -1.847759067047173
        intercept = 10.168018059896724
        ref0      = 5.50289172
        ref1      = 8.91421356
        mins      = [0] * Setting.n_var
        maxs      = [1] * Setting.n_var
        
    elif problem == "DEB2DK":
        num_obj  = 2
        problem  = DEB2DK().objective()
        observer = trieste.objectives.utils.mk_observer(problem)
        if Setting.knees == 1:
            hv_best   = np.array([2.82571712, 2.83115024])
            dl_best   = np.array([2.82571712, 2.83115024])
            slope     = -1.0
            intercept = 8.5
            ref0      = 8.5
            ref1      = 8.5
        elif Setting.knees == 2:
            ref0         = 8.0
            ref1         = 8.0
            ground_truth = 28.798335754990344
        elif Setting.knees == 3:
            ref0         = 7.83333333333
            ref1         = 7.83333333333
            ground_truth = 28.680189723910743
        
        mins = [0] * Setting.n_var
        maxs = [1] * Setting.n_var
        
    elif problem == "DTLZ7":
        num_obj   = 2
        problem   = DTLZ7().objective()
        observer  = trieste.objectives.utils.mk_observer(problem)
        hv_best   = np.array([0.22810017, 3.5809773 ])
        dl_best   = np.array([0.83382007, 2.33236865])
        ref0      = 0.85940549
        ref1      = 4.00000001
        slope     = -1.9699613805216913
        intercept = 4.000000007965181
        mins      = [0] * Setting.n_var
        maxs      = [1] * Setting.n_var
        
    elif problem == "CKP":
        num_obj      = 2
        problem      = ckp
        observer     = trieste.objectives.utils.mk_observer(problem)
        hv_best      = np.array([2.99643514, 3.01173714])
        dl_best      = np.array([2.91496276, 3.07755593])
        ref0         = 7
        ref1         = 6
        slope        = -0.8571428571428572
        intercept    = 6.000000000000002
        ground_truth = 10.277846179363369
        mins         = [0] * Setting.n_var
        maxs         = [1] * Setting.n_var
        if Setting.knees == 2:
            ref0 = 6.5
            ref1 = 5.5
            
    elif problem == "ZDT3":
        num_obj   = 2
        problem   = zdt3
        observer  = trieste.objectives.utils.mk_observer(problem)
        hv_best   = np.array([0.99841826, 1.001583  ])
        dl_best   = np.array([2.91496276, 3.07755593])
        ref0      = 0.85182557
        ref1      = 1.0
        slope     = -1
        intercept = 4
        mins      = [0] * Setting.n_var
        maxs      = [1] * Setting.n_var
    
    elif problem == "DEB3DK":
        num_obj  = 3
        problem  = DEB3DK().objective()
        observer = trieste.objectives.utils.mk_observer(problem)
        ref0     = 23.75
        ref1     = 9.5
        ref2     = 9.5
        hv_best  = np.array([2.85119524, 2.82749419, 3.52386297])
        mins     = [0] * Setting.n_var
        maxs     = [1] * Setting.n_var
        
    elif problem == "CBEAM":
        num_obj   = 2
        problem   = Cbeam().objective()
        observer  = trieste.objectives.utils.mk_observer(problem)
        hv_best   = np.array([224.30994738, 237.40184873])
        dl_best   = np.array([224.30994738, 237.40184873])
        slope     = -1.1064711456446412
        intercept = 1407.077137258377
        ref0      = 1224.
        ref1      = 1243.76199616
        mins      = [0] * Setting.n_var
        maxs      = [1] * Setting.n_var
            
    elif problem == "CYCLONE":
        mins      = [0.4, 0.14, 0.4, 3.0, 1.0, 0.4, 0.2]
        maxs      = [0.7, 0.4, 0.75, 7.0, 2.0, 2.0, 0.4]
        num_obj   = 2
        n_var     = 7
        knees     = 1
        problem   = Cyclone().objective()
        observer  = trieste.objectives.utils.mk_observer(problem)
        hv_best   = np.array([0.31750461, 0.13021828])
        dl_best   = np.array([0.31750461, 0.13021828])
        ref0      = 2.47591462
        ref1      = 2.21129756
        slope     = -0.8671698713971665
        intercept = 1.9215192704886834
    
    else:
        raise NotImplementedError("Please select the correct problem")
    
    search_space = trieste.space.Box(mins, maxs)

    num_initial_points   = (Setting.n_var + 1) * (Setting.n_var + 2) / 2
    initial_query_points = search_space.sample_halton(num_initial_points)
    dataset              = observer(initial_query_points)

    model   = build_stacked_independent_objectives_model(dataset, num_obj, search_space)
    bo      = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    regrets_hv  = []
    regrets_d2l = []

    if method == "HVKTS":
        acq  = HVThompson()
        rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq)
    elif method == "HVKTS-EA":
        rule = HVNSGARule()
    elif method == "EHVI":
        acq  = trieste.acquisition.ExpectedHypervolumeImprovement()
        rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq)
    elif method == "HV-KNEE":
        acq   = ExtremaFinder()
        rule  = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq)
        acqb  = HvKnee()
        ruleb = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acqb)

    reps = kwargs["reps"]
    if method == "hv-knee":
        reps = int(reps/2)

    if Setting.knees == 1 and num_obj == 2:
        regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
        regrets_d2l.append(calculate_regret_d2l(slope, intercept, dl_best, dataset))
    elif num_obj == 3:
        regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
    else:
        regrets_hv.append(calculate_regret_hv_multiple(ref0, ref1, dataset, ground_truth))

    i = 0
    while i < reps:
        try:
            result  = bo.optimize(1, dataset, model, acquisition_rule = rule)
            dataset = result.try_get_final_dataset()
            if Setting.knees == 1 and num_obj == 2:
                regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
                regrets_d2l.append(calculate_regret_d2l(slope, intercept, dl_best, dataset))
            elif num_obj == 3:
                regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
            else:
                regrets_hv.append(calculate_regret_hv_multiple(ref0, ref1, dataset, ground_truth))
            i += 1
        except:
            model   = build_stacked_independent_objectives_model(dataset, num_obj, search_space)
        
        if method == "hv-knee":
            result  = bo.optimize(1, dataset, model, acquisition_rule = ruleb)
            dataset = result.try_get_final_dataset()
            if Setting.knees == 1 and num_obj == 2:
                regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
                regrets_d2l.append(calculate_regret_d2l(slope, intercept, dl_best, dataset))
            elif num_obj == 3:
                regrets_hv.append(calculate_regret_hv(ref0, ref1, hv_best, dataset, ref2))
            else:
                regrets_hv.append(calculate_regret_hv_multiple(ref0, ref1, dataset, ground_truth))

    return regrets_d2l, regrets_hv, model, dataset