import numpy      as np
import tensorflow as tf
import trieste
import gpflow

from config import Setting

def hyperVolume(point: np.ndarray, reference: np.ndarray):
    """Calculates the hypervolume between a point and the reference point
    
    Parameters
    ----------
    point: np.ndarray
        A point in the pareto front
    reference: np.ndarray
        The reference point

    Returns
    -------
    int
        The calculated hypervolume 
    """
    hyper_volume = 1
    for dim in range(len(point)):
        hyper_volume *= np.abs(point[dim] - reference[dim])

    return hyper_volume

def sortParetoFront(pareto_front: np.ndarray):
    """Sorts the pareto front based on the first objective
    
    Parameters
    ----------
    pareto_front: np.ndarray
        The points that are in the pareto front

    Returns
    -------
    np.ndarray
        Sorted pareto front
    """
    out = pareto_front.shape[-1]
    return np.array(sorted(pareto_front.tolist(), key = lambda x: x[0])).reshape(-1,out)

def lineFinder(extreme1: np.ndarray, extreme2: np.ndarray):
    """Calculates the slope and the intercept of the line that connects two points
    
    Parameters
    ----------
    extreme1: np.ndarray
        First point
    extreme2: np.ndarray
        Second point

    Returns
    -------
    (float, float)
        Slope and Intercept of the line that connects the two extremes
    """
    slope     = (extreme1[1] - extreme2[1]) / (extreme1[0] - extreme2[0])
    intercept = extreme1[1] - slope * extreme1[0]

    return slope, intercept


def distanceToLine(point: np.ndarray, slope: int, intercept: int):
    """Calculates the distance between a point and a line
    
    Parameters
    ----------
    point: np.ndarray
        The point
    slope: float
        Slope of the line
    intercept: float
        Intercept of the line

    Returns
    -------
    float
        The calculated distance between the specified point and line
    """
    prep_slope     = -1 / slope
    prep_intercept = point[1] - prep_slope * point[0]

    intersection_x = (prep_intercept - intercept) / (slope - prep_slope)
    intersection_y = slope * intersection_x + intercept

    distance = ((intersection_x - point[0])**2 + (intersection_y - point[1])**2)**0.5
    if point[0] > intersection_x:
        distance *= -1
    return distance

def build_stacked_independent_objectives_model(data, num_output, search_space):
    gprs = []
    for idx in range(num_output):
        single_obj_data = trieste.data.Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        kernel          = gpflow.kernels.RBF(lengthscales = np.random.rand(Setting.n_var))
        gpr             = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
        gprs.append((trieste.models.gpflow.GaussianProcessRegression(gpr), 1))

    return trieste.models.TrainableModelStack(*gprs)

def calculate_regret_hv(ref0, ref1, best, dataset, ref2 = None):
    if ref2 == None:
        hvs           = (ref1 - dataset.observations.numpy()[:,1]) * (ref0 - dataset.observations.numpy()[:,0])
        mask          = tf.logical_and(dataset.observations.numpy()[:,0] < ref0 , dataset.observations.numpy()[:,1] < ref1)
        hvs           = tf.where(mask, hvs, 0)
        known_knee_hv = hyperVolume(best, np.array([ref0,ref1]))

    else:
        hvs           = (ref0 - dataset.observations.numpy()[:,0]) * (ref1 - dataset.observations.numpy()[:,1]) * (ref2 - dataset.observations.numpy()[:,2])
        mask1         = tf.logical_and(dataset.observations.numpy()[:,0] < ref0 , dataset.observations.numpy()[:,1] < ref1)
        mask          = tf.logical_and(dataset.observations.numpy()[:,2] < ref2 , mask1)
        hvs           = tf.where(mask, hvs, 0)
        known_knee_hv = hyperVolume(best, np.array([ref0,ref1,ref2]))
    
    return known_knee_hv - tf.reduce_max(hvs).numpy()

def calculate_regret_d2l(slope, intercept, best, dataset):
    prep_slope      = -1 / slope
    prep_intercept  = dataset.observations.numpy()[:,1] - prep_slope * dataset.observations.numpy()[:,0]
    intersection_x  = (prep_intercept - intercept) / (slope - prep_slope)
    intersection_y  = slope * intersection_x + intercept
    distances       = ((intersection_x - dataset.observations.numpy()[:,0])**2 + (intersection_y - dataset.observations.numpy()[:,1])**2)**0.5
    mask            = dataset.observations.numpy()[:,0] < intersection_x
    distances       = tf.where(mask, distances, -1 * distances)
    known_knee_dist = distanceToLine(best, slope, intercept)
    return known_knee_dist - tf.reduce_max(distances).numpy()

def calculate_regret_hv_multiple(ref0, ref1, dataset, ground_truth):
    largest_volume = 0
    pareto_front   = list()
    pareto_front_n = sortParetoFront(trieste.acquisition.multi_objective.non_dominated(dataset.observations)[0].numpy()) 
    for point in pareto_front_n:
        if point[0] < ref0 and point[1] < ref1:
            pareto_front.append(point)
    pareto_front = np.array(pareto_front)
    
    if Setting.knees == 3:
        for index1, point1 in enumerate(pareto_front):
            for index2, point2 in enumerate(pareto_front[index1:]):
                for point3 in enumerate(pareto_front[index1 + index2:]):
                    volume1 = hyperVolume(point = point1, reference = np.array([ref0, ref1]))
                    volume2 = hyperVolume(point = point2, reference = np.array([ref0, point1[1]]))
                    volume3 = hyperVolume(point = point3, reference = np.array([ref0, point2[1]]))
                    volume  = volume1 + volume2 + volume3
                    if volume > largest_volume:
                        largest_volume = volume

    if Setting.knees == 2:
        for index1, point1 in enumerate(pareto_front):
            for index2, point2 in enumerate(pareto_front[index1:]):
                volume1 = hyperVolume(point = point1, reference = np.array([ref0, ref1]))
                volume2 = hyperVolume(point = point2, reference = np.array([ref0, point1[1]]))
                volume  = volume1 + volume2
                if volume > largest_volume:
                    largest_volume = volume
    
    return ground_truth - largest_volume
