import cv2
import numpy as np

augmentations = [
        'additive_gaussian_noise',
        'additive_speckle_noise',
        'random_brightness',
        'random_contrast',
        'additive_shade',
        'motion_blur'
]

def additive_gaussian_noise(image, stddev_range=[0.0, 0.06]):
    stddev = np.random.uniform(*stddev_range)
    image += np.random.normal(loc=0.0, scale=stddev, size=image.shape)
    image = np.clip(image, a_min=0.0, a_max=1.0)
    return image

def additive_speckle_noise(image, prob_range=[0.0, 0.005]):
    prob = np.random.uniform(*prob_range)
    sample = np.random.uniform(size=image.shape)
    image[sample < prob] = 0.0
    image[sample > (1.0 - prob)] = 1.0
    return image

def random_brightness(image, max_abs_change=0.2):
    delta = np.random.uniform(-max_abs_change, max_abs_change)
    return np.clip(image + delta, a_min=0.0, a_max=1.0)

def random_contrast(image, strength_range=[0.5, 1.5]):
    mean = image.mean()
    strength = np.random.uniform(*strength_range)
    image = (image - mean) * strength + mean
    return np.clip(image, a_min=0.0, a_max=1.0)

def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):

    min_dim = min(image.shape[:2]) / 4
    mask = np.zeros(image.shape[:2], np.float32)
    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, image.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 1.0, -1)

    transparency = np.random.uniform(*transparency_range)
    kernel_size = np.random.randint(*kernel_size_range)
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    shaded = image * (1 - transparency * mask)
    return np.clip(shaded, 0, 1.0)

def motion_blur(image, max_kernel_size=10):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.0
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.0*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    image = cv2.filter2D(image, -1, kernel)
    return image
