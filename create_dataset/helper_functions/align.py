import cv2
import math
import multiprocessing
import numpy as np
from scipy.optimize import minimize
import signal
from sklearn.metrics import mutual_info_score
from scipy.ndimage import gaussian_filter

from helper_functions.disp import *
from helper_functions.utils import *

def warp_image(image, transform, height, width):
    '''
    Warp an image using a specified transform (affine or perspective).
    Border values are filled with -1.0

    Parameters
    ----------
    image: numpy.array
    transform: numpy.array
        The current transformation matrix (2x3 affine, 3x3 perspective)
    optical : numpy.array
        Greyscale optical image
    height : int
        Target image height
    width : int
        Target image width

    Returns
    -------
    warped : numpy.array
        Warped image, of specified size (height, width)
    '''
    if transform.shape == (3,3):
        warped = cv2.warpPerspective(image,
                                     np.linalg.inv(transform),
                                     (width, height),
                                     borderValue = -1.0)

    elif transform.shape == (2,3):
        transform_inv = cv2.invertAffineTransform(transform)
        warped = cv2.warpAffine(image,
                                transform_inv,
                                (width, height),
                                borderValue = -1.0)
    else:
        raise ValueError('Unknown transformation shape: ', transform.shape)

    return warped

def mutual_information_2d(x, y, sigma=5, bins = 100, normalized = False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    bins: int
        number of bins for the histogram
    normalized: boolean
        indicates if the mi is normalized
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    jh = np.histogram2d(x, y, bins=(bins, 2*bins))[0]

    # smooth the jh with a gaussian filter of given sigma
    if sigma > 0:
        gaussian_filter(jh, sigma=sigma, mode='constant',
                                     output=jh)

    # compute marginal histograms
    jh = jh + np.finfo(float).eps
    sh = np.sum(jh)

    jh = jh / sh

    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[1]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[0], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi

def calculate_negative_mutual_information(transform, optical, thermal, init_transform, bins,
                                          regularize = False, normalized_mi = False, smoothing_sigma = 0):
    '''
    Calculate the negative mutual information between the warped optical image and the thermal image.

    Parameters
    ----------
    transform: numpy.array
        The current transformation matrix
    optical : numpy.array
        Greyscale optical image
    thermal : numpy.array
        Greyscale thermal image
    init_transform: numpy.array
        The initial transformation matrix to start the refinement
    bins : int
        The number of bins used to compute the mutual information
    regularize : bool
        If true the norm of the difference between the current and initial
        transformation is added to the objective
    normalized_mi : bool
        Indicates if the mutual information should be normalized
    smoothing_sigma : flaot
        Sigma of the gaussian smoothing filter to smooth the 2d histogram

    Returns
    -------
    mi : float
        The negative mutual information (including the regularization term if set by the input)
    '''

    # warp the image
    h, w = thermal.shape

    # warp the optical image
    if len(transform) == 4:
        transform = np.array([[transform[1]*np.cos(transform[0]), transform[1]*np.sin(transform[0]), transform[2]],
                              [-transform[1]*np.sin(transform[0]), transform[1]*np.cos(transform[0]), transform[3]]])
    else:
        transform = transform.reshape(init_transform.shape)

    optical_warped = warp_image(optical, transform, h, w)

    # compute the mutual information
    mi = mutual_information_2d(optical_warped.ravel(),
                               thermal.ravel(),
                               bins = bins,
                               normalized = normalized_mi,
                               sigma = smoothing_sigma)

    if regularize:
        return -mi + np.linalg.norm(init_transform - transform)
    else:
        return -mi

def refine_alignment(optical, thermal, init_transform, decompose_transformation, regularize,
                     bins = 256, normalized_mi = False, smoothing_sigma = 0):
    '''
    Refine the alignment by maximizing the mutual information between the optical and thermal image.
    Either an affine (2x3 matrix) or perspective (3x3) transform can be refined.

    Parameters
    ----------
    optical : numpy.array
        Greyscale optical image
    thermal : numpy.array
        Greyscale thermal image
    init_transform: numpy.array
        The initial transformation matrix to start the refinement
    decompose_transformation : bool
        If true the affine transformation only contains:
        scale, one rotation and two translations
    regularize : bool
        If true the norm of the difference between the current and initial
        transformation is added to the objective
    bins : int
        The number of bins used to compute the mutual information
    normalized_mi : bool
        Indicates if the mutual information should be normalized
    smoothing_sigma : flaot
        Sigma of the gaussian smoothing filter to smooth the 2d histogram

    Returns
    -------
    transform : numpy.array
        The refined transformation matrix
    success : bool
        Indicates if the optimization algorithm was successful
    '''

    if decompose_transformation and not (init_transform.shape == (2,3)):
        decompose_transformation = False

    if decompose_transformation:
        angle = np.arctan2(init_transform[0,1],init_transform[0,0])
        init_transform = np.array([angle,                                # rotation
                                   init_transform[0,0] / np.cos(angle),  # scale
                                   init_transform[0,2],                  # dx
                                   init_transform[1,2]])                 # dy

    res = minimize(calculate_negative_mutual_information,
                   init_transform,
                   args = (optical, thermal, init_transform, bins, regularize, normalized_mi, smoothing_sigma),
                   method = 'Nelder-Mead',
                   options = {'adaptive': False, 'xatol': 1e-6, 'fatol': 1e-6}
                   )

    if decompose_transformation:
        transform = np.array([[res.x[1]*np.cos(res.x[0]), res.x[1]*np.sin(res.x[0]), res.x[2]],
                              [-res.x[1]*np.sin(res.x[0]), res.x[1]*np.cos(res.x[0]), res.x[3]]])
    else:
        transform = res.x.reshape(init_transform.shape)

    return transform, res.success

def wrapper_refine_alignement(input):
    '''
    Wrapper for the refine_alignment function reducing the input and output to a dictionary.

    Parameters
    ----------
    input : dictionary
        Dictionary with the following keys:
            optical: Optical image
            thermal: Thermal image
            t_init: Initial transformation matrix
            decompose: Decompose the transformation in case of an affine transformation
            n_bin: Number of bins used to compute the mutual information

    Returns
    -------
    res : dictionary
        Dictionary with the following keys:
            name: Name of the refinement algorithm to distinguish different settings
            transformation: Refined transformation matrix
            solver_success: Indicates if the solver converged
            type: String indicating if the transformation is affine or perspective
    '''
    res = dict()
    res['transformation'], res['solver_success'] = refine_alignment(input['optical'],
                                                                    input['thermal'],
                                                                    input['t_init'],
                                                                    input['decompose'],
                                                                    False,
                                                                    input['n_bin'],
                                                                    input['params']['alignment/normalized_mi'],
                                                                    input['params']['alignment/smoothing_sigma'])
    res['name'] = 'bin' + str(input['n_bin'])
    if input['t_init'].shape == (2, 3):
        res['name'] += '_affine'
        res['type'] = 'affine'
    else:
        res['type'] = 'perspective'

    if input['params']['alignment/normalized_mi']:
        res['name'] += '_normalized'

    res['name'] += '_s' + str(input['params']['alignment/smoothing_sigma'])

    # warp the image
    h, w = input['thermal'].shape[:2]
    res['warped_image'] = warp_image(input['optical_rgb'], res['transformation'], h, w)

    # check if the transformations are valid
    if res['type'] == 'perspective':
        res['valid'], res['mi'] = check_perspective_transformation(input['t_init'],
                                                                   res['transformation'],
                                                                   input['optical'],
                                                                   input['thermal'],
                                                                   input['params'],
                                                                   input['verbose'])
    elif res['type'] == 'affine':
        res['valid'], res['mi'] = check_affine_transformation(input['t_init'],
                                                              res['transformation'],
                                                              input['optical'],
                                                              input['thermal'],
                                                              input['params'],
                                                              input['verbose'])

    return res

def check_perspective_transformation(init_transformation, transformation, optical, thermal, params, verbose = False):
    '''
    Check if the perspective transformation matrix is valid by decomposing the transformation and checking the
    rotations and translations individually.

    Parameters
    ----------
    init_transform: numpy.array
        The initial transformation matrix to start the refinement
    transform: numpy.array
        The current transformation matrix
    optical : numpy.array
        Greyscale optical image
    thermal : numpy.array
        Greyscale thermal image
    params : dict
        Dictionary containing the limits for the transformation
    verbose : bool
        Indicates if additional information should be printed to the console

    Returns
    -------
    valid : bool
        Indicates if the transformation is in the specified bounds
    mi : float
        The negative mutual information (including the regularization term if set by the input)
    '''

    # decompose the transformation into a rotation and translation matrix
    num, Rs_init, Ts_init, Ns  = cv2.decomposeHomographyMat(init_transformation,
                                                            np.array([[1.0,0,0],[0.0,1.0,0],[0.0,0.0,1.0]]))
    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(transformation,
                                                  np.array([[1.0,0,0],[0.0,1.0,0],[0.0,0.0,1.0]]))

    # compute the difference of the rotations and translations
    diff_rot = 180.0 / math.pi * (rotationMatrixToEulerAngles(Rs_init[2]) - rotationMatrixToEulerAngles(Rs[2]))
    diff_translation = Ts_init[2] - Ts[2]

    # compute the difference in the MI score
    init_mi = calculate_negative_mutual_information(init_transformation,
                                                    optical,
                                                    thermal,
                                                    init_transformation,
                                                    100,
                                                    normalized_mi = params['alignment/normalized_mi'],
                                                    smoothing_sigma = params['alignment/smoothing_sigma'])
    new_mi = calculate_negative_mutual_information(transformation,
                                                   optical,
                                                   thermal,
                                                   init_transformation,
                                                   100,
                                                   normalized_mi = params['alignment/normalized_mi'],
                                                   smoothing_sigma = params['alignment/smoothing_sigma'])
    diff_mi = (init_mi - new_mi)

    # check based on the bounds if the transformation is valid
    valid = ((abs(diff_mi) < params['alignment/check/both/max_diff_mi']) and
             (abs(diff_translation[0]) < params['alignment/check/both/max_diff_dx']) and
             (abs(diff_translation[1]) < params['alignment/check/both/max_diff_dy']) and
             (abs(diff_translation[2]) < params['alignment/check/perspective/max_diff_dz']) and
             (abs(diff_rot[0]) < params['alignment/check/perspective/max_diff_roll']) and
             (abs(diff_rot[1]) < params['alignment/check/perspective/max_diff_pitch']) and
             (abs(diff_rot[2]) < params['alignment/check/perspective/max_diff_yaw']))

    # check if warp returns a valid image
    if params['alignment/check/invalid_pixels']:
        h, w = thermal.shape
        warped = warp_image(optical, transformation, h, w)
        if warped.min() == -1.0:
            # If any border values were added during the transform
            valid = False

    if verbose:
        print('transformation matrix:')
        print(transformation)
        print('valid: ', valid, ', mi: ', new_mi)
        print('diff rotations: ', diff_rot, ', diff translations: ',np.transpose(diff_translation))

    return valid, new_mi

def check_affine_transformation(init_transformation, transformation, optical, thermal, params, verbose = False):
    '''
    Check if the affine transformation matrix is valid by decomposing the transformation and checking the
    rotations and translations individually.

    Parameters
    ----------
    init_transform: numpy.array
        The initial transformation matrix to start the refinement
    transform: numpy.array
        The current transformation matrix
    optical : numpy.array
        Greyscale optical image
    thermal : numpy.array
        Greyscale thermal image
    params : dict
        Dictionary containing the limits for the transformation
    verbose : bool
        Indicates if additional information should be printed to the console

    Returns
    -------
    valid : bool
        Indicates if the transformation is in the specified bounds
    mi : float
        The negative mutual information (including the regularization term if set by the input)
    '''

    # (rotations and shear)
    rot = np.arctan2(transformation[0,1], transformation[0,0])
    rot_init = np.arctan2(init_transformation[0,1], init_transformation[0,0])
    shear = np.arctan2(transformation[1,1], transformation[1,0]) - math.pi * 0.5 - rot
    shear_init = np.arctan2(init_transformation[1,1], init_transformation[1,0]) - math.pi * 0.5 - rot_init

    # scale
    s_x = np.sqrt(transformation[1,0]**2+transformation[0,0]**2)
    s_x_init = np.sqrt(init_transformation[1,0]**2+init_transformation[0,0]**2)
    s_y = np.sqrt(transformation[1,1]**2+transformation[0,1]**2) * np.cos(shear)
    s_y_init = np.sqrt(init_transformation[1,1]**2+init_transformation[0,1]**2) * np.cos(shear_init)

    # translation
    diff_translation = init_transformation[:,2] - transformation[:,2]

    # compute the difference in the MI score
    init_mi = calculate_negative_mutual_information(init_transformation,
                                                    optical,
                                                    thermal,
                                                    init_transformation,
                                                    100,
                                                    normalized_mi = params['alignment/normalized_mi'],
                                                    smoothing_sigma = params['alignment/smoothing_sigma'])
    new_mi = calculate_negative_mutual_information(transformation,
                                                   optical,
                                                   thermal,
                                                   init_transformation,
                                                   100,
                                                   normalized_mi = params['alignment/normalized_mi'],
                                                   smoothing_sigma = params['alignment/smoothing_sigma'])

    # check based on the bounds if the transformation is valid
    valid = ((abs(init_mi - new_mi) < params['alignment/check/both/max_diff_mi']) and
             (abs(diff_translation[0]) < params['alignment/check/both/max_diff_dx']) and
             (abs(diff_translation[1]) < params['alignment/check/both/max_diff_dy']) and
             (abs(s_x - s_x_init) < params['alignment/check/affine/max_diff_scale']) and
             (abs(s_y - s_y_init) < params['alignment/check/affine/max_diff_scale']) and
             (abs(rot - rot_init) < math.radians(params['alignment/check/affine/max_diff_rotation'])) and
             (abs(shear - shear_init) < math.radians(params['alignment/check/affine/max_diff_shear'])))

    # check if warp returns a valid image
    if params['alignment/check/invalid_pixels']:
        h, w = thermal.shape
        warped = warp_image(optical, transformation, h, w)
        if warped.min() == -1.0:
            valid = False

    if verbose:
        print('transformation matrix:')
        print(transformation)
        print('valid: ', valid, ',mi: ', new_mi, ', diff rot: ', rot - rot_init, ', diff shear: ', shear - shear_init)
        print('diff scale x: ', s_x - s_x_init, ', diff scale y: ', s_y - s_y_init, ', diff translations: ', diff_translation)

    return valid, new_mi

def align_images(optical,
                 thermal,
                 init_transform,
                 params,
                 counter,
                 verbose = False,
                 show_results = False,
                 filter_images = False):
    '''
    Align the images using mutual information starting from an initial guess.
    Different settings (bin sizes) for the mutual information, as well as
    affine and perspective transformations are used.
    It is checked if each transformation is valid and then the best one 
    based on mutual information is picked.
    All images warped with a valid transformation are returned.

    Parameters
    ----------
    optical: numpy.array
        Optical image, bgr
    thermal: numpy.array
        Thermal image, greyscale
    verbose: bool
        Indicates if extra information should be printed to the terminal
    show_results: bool
        Show GIF's of the aligned images. Blocks until the n key is pressed

    Returns
    -------
    best_warped_image : numpy.array
        Based on the mutual information metric the best warped optical image
    valid_warped_images : list(numpy.array)
        All the optical images warped by a valid transformation
    '''

    # initial guess for the transform (determined manually)
    decompose_transformation = params['alignment/decomposed_transformation']

    # convert the optical image to grayscale
    optical_greyscale = cv2.cvtColor(optical.astype(np.float32) / 255.0, cv2.COLOR_BGR2GRAY)

    if filter_images:
        thermal = cv2.GaussianBlur(thermal,(params['alignment/filter_size'],params['alignment/filter_size']),0)
        optical_greyscale = cv2.GaussianBlur(optical_greyscale,(params['alignment/filter_size'],params['alignment/filter_size']),0)

    if params['alignment/run_optimization']:
        inputs = []
        for bins in params['alignment/bin_sizes']:
            inputs.append({'n_bin': bins, 't_init': init_transform, 'optical': optical_greyscale, 'optical_rgb': optical,
                           'thermal': thermal, 'decompose': False, 'params': params, 'verbose': verbose})

        if params['alignment/use_multiprocess']:
            def init_worker():
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            p = multiprocessing.Pool(len(inputs), init_worker)
            try:
                ret = p.map_async(wrapper_refine_alignement, inputs)
                alignments = ret.get(timeout=params['alignment/optimization_timeout'])
            except multiprocessing.TimeoutError:
                p.terminate()
                p.join()
                return False, True, None, None, None, counter
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                print('Received KeyboardInterrupt, terminating program')
                exit()
            p.close()
            p.terminate()
            p.join()
        else:
            alignments = [wrapper_refine_alignement(i) for i in inputs]
    else:
        # no refined alignments
        alignments = []

    if params['alignment/accept_init']:
        # warp the initial image
        h, w = thermal.shape[:2]

        init_warped = warp_image(optical, init_transform, h, w)

        # process the initial alignment
        mi = {}
        for bins in params['alignment/bin_sizes']:
            mi[str(bins)] = calculate_negative_mutual_information(init_transform,
                                                                  optical_greyscale,
                                                                  thermal,
                                                                  init_transform,
                                                                  bins)
        valid_mi = [mi]
        types = ['init']
        valid_transforms = [init_transform]
        valid_warped_images = [init_warped]
    else:
        valid_mi = []
        types = []
        valid_transforms = []
        valid_warped_images = []

    for elem in alignments:
        # consider alignment if the solver converged and the transformation is valid
        if elem['valid'] and elem['solver_success']:
            mi = {}
            for bins in params['alignment/bin_sizes']:
                mi[str(bins)] = calculate_negative_mutual_information(elem['transformation'],
                                                                     optical_greyscale,
                                                                     thermal,
                                                                     init_transform,
                                                                     bins)

            valid_mi.append(mi)
            types.append(elem['name'])
            valid_transforms.append(elem['transformation'])
            valid_warped_images.append(elem['warped_image'])


    if len(valid_mi) < 1:
        # no valid solution
        return False, False, None, None, None, counter
    else:
        # find the best solution
        if params['alignment/ranking_method'] == 'sum':
            summed_mi = []
            for mi in valid_mi:
                tmp = 0.0
                for key in mi.keys():
                    tmp += mi[key]
                summed_mi.append(tmp)

            best_idx = np.argmin(summed_mi)

        elif params['alignment/ranking_method'] == 'order':
            reordered_mi = {}
            for key in valid_mi[0].keys():
                reordered_mi[key] = []

            for mi in valid_mi:
                for key in mi.keys():
                    reordered_mi[key].append(mi[key])

            ranking = np.zeros((len(valid_mi)))
            for key in reordered_mi.keys():
                ranking += np.array(reordered_mi[key]).argsort()

            best_idx = np.argmin(ranking)

        else:
            raise ValueError('Unknown ranking_method')

    best_warped_image = valid_warped_images[best_idx]
    best_transform = valid_transforms[best_idx]
    best_type = types[best_idx]

    # update the statistics
    if best_type in counter:
        counter[best_type] += 1
    else:
        counter[best_type] = 1
    counter['total'] += 1

    if verbose:
        print('Best alignment: ' + str(best_type))

    if show_results:
        show_image_gifs(valid_warped_images, len(valid_warped_images) * [thermal])

    return True, False, best_warped_image, best_transform, valid_warped_images, counter

def manual_aligmnent(optical, thermal):
    '''
    Estimate the initial transformation using manually specified keypoints.

    Parameters
    ----------
    optical : numpy.array
        Greyscale optical image
    thermal : numpy.array
        Greyscale thermal image
    '''

    # manually selected keypoints
    ir_points = np.float32([
        [468,324],
        [509,243],
        [552,417],
        [574,412],
        [625,384],
        [630,427],
        [634,447],
        [90,86],
        [172,51],
        [194,41],
        [160,41],
        [128,123],
        [208,182],
        [205,182],
        [17,203],
        [24,193],
        [86,298],
        [88,289],
        [128,261],
        [138,257],
        [291,173],
        [294,158],
        [268,162],
        [278,191],
        [332,10],
        [261,486],
        [277,366],
        [316,369],
        [362,389],
        [402,402],
        [430,383],
        [429,215],
        [486,133],
        [551,13]])

    opt_points = np.float32([
        [587,321],
        [629,245],
        [669,409],
        [688,404],
        [740,378],
        [744,418],
        [747,438],
        [232,93],
        [309,61],
        [332,49],
        [297,51],
        [266,128],
        [341,183],
        [338,163],
        [161,203],
        [167,194],
        [225,293],
        [226,283],
        [265,258],
        [275,254],
        [421,175],
        [424,161],
        [399,165],
        [408,193],
        [463,10],
        [387,474],
        [405,357],
        [441,362],
        [486,381],
        [523,393],
        [550,377],
        [554,215],
        [608,138],
        [674,23]])

    # compute the transformation
    h, w = thermal.shape[:2]
    transform = cv2.estimateRigidTransform(ir_points, opt_points, False)
    homography, mask = cv2.findHomography(ir_points, opt_points, cv2.LMEDS)

    inverse_tf = cv2.invertAffineTransform(transform)
    warped_affine = cv2.warpAffine(optical, inverse_tf, (w, h))
    warped_homography = cv2.warpPerspective(optical, np.linalg.inv(homography), (w, h))

    print('Homography:')
    print(homography)
    print('Affine Transformation:')
    print(transform)

    # show the warped images
    show_image_gifs([warped_affine, warped_homography],[thermal, thermal])
