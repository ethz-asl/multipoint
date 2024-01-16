import argparse
import collections
import cv2
import logging
import math
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import yaml

from helper_functions.align import *
from helper_functions.disp import *
from helper_functions.utils import *

# --------------------------------------------------------------------------------------------------
# ---------------------- IMAGE ALIGNER -------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
class ImageAligner():
    '''
    Class to iterate over a image pairs and align them using mutual information.
    '''

    def __init__(self, input_dir, output_dir, config_file):
        '''
        Initializer
        
            Parameters
        ----------
        input_dir : string
            Directory containing the preprocessed image pairs
        output_dir : string
            Directory where the processed images will be stored
        bagfile : string
            Name of the bagfile to process
        config_file : string
            Yaml file specifying the alignment process
        '''

        self.__input_dir = input_dir
        self.__output_dir = output_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        with open(config_file, 'rt') as fh:
            self.__params = yaml.safe_load(fh)

        self.__optical_filenames = [f for f in os.listdir(input_dir)
                                    if os.path.isfile(os.path.join(input_dir, f)) and 'optical' in f]

        print('Number of pairs: ' + str(len(self.__optical_filenames)))

        self.__counter = dict()
        self.__counter['total'] = 0

        logging.basicConfig(filename=str(Path(output_dir, 'failed.log')),
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(message)s')

        # initial guess for the transform (determined manually) TODO move to calibration file
        with open(str(Path(input_dir, 'initial_transform.yaml')), 'rt') as fh:
            init_params = yaml.safe_load(fh)
            self.__init_transform_perspective = np.array(init_params['perspective'])
            self.__init_transform_affine = np.array(init_params['affine'])

            # initialize the averaged transformation with the initial guess
            num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(self.__init_transform_perspective,
                                                          np.array([[1.0,0,0],[0.0,1.0,0],[0.0,0.0,1.0]]))

            # compute the difference of the rotations and translations
            self.__averaged_rotation = rotationMatrixToEulerAngles(Rs[2])
            self.__averaged_translation = Ts[2]
            self.__averaged_n_measurements = 1.0

    def run(self):
        '''
        Run the alignment process by iterating through the image pairs and
        refine the alignment from a predetermined transformation using
        mutual information.
        '''

        # create directories if required
        if self.__params['save_aligned_images']:
            out_path_aligned_best = Path(self.__output_dir, 'aligned', 'best')
            out_path_aligned_all = Path(self.__output_dir, 'aligned', 'all')
    
            if not os.path.isdir(str(out_path_aligned_best)):
                os.makedirs(str(out_path_aligned_best))
            if not os.path.isdir(str(out_path_aligned_all)):
                os.makedirs(str(out_path_aligned_all))

        # iterate over the images
        with tqdm(total=len(self.__optical_filenames)) as pbar:
            for optical_name in self.__optical_filenames:
                # get the image paths
                index = optical_name.split('_')[0]
                optical_path = Path(self.__input_dir, optical_name)
                thermal_path = Path(self.__input_dir, index + '_thermal.png')
                thermal_raw_path = Path(self.__input_dir, index + '_thermal_raw.png')

                # load the images
                optical = cv2.imread(str(optical_path), -1)
                thermal = cv2.imread(str(thermal_path), -1)
                thermal_raw = cv2.imread(str(thermal_raw_path), -1)

                # align the images
                if self.__params['perspective']:
                    t_init = np.copy(self.__init_transform_perspective)
                else:
                    t_init = np.copy(self.__init_transform_affine)

                transformation = np.copy(t_init)

                if self.__params['alignment_method'] == 'mi':
                    (success, solver_fail, best_warped, transformation,
                        valid_warped) = self.align_images_mutual_information(optical,
                                                                             thermal,
                                                                             thermal_raw,
                                                                             t_init.copy())
                else:
                    raise ValueError('Unkown alignment method: ' + self.__params['alignment_method'])

                if success:
                    # update the averaged transformation
                    self.update_average_transformation(transformation)

                    if self.__params['save_aligned_images']:
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_optical.png')),
                                    best_warped)
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_thermal_raw.png')),
                                    thermal_raw)
                        cv2.imwrite(str(Path(out_path_aligned_best, index + '_thermal.png')),
                                    thermal)

                        for i, image in enumerate(valid_warped):
                            cv2.imwrite(str(Path(out_path_aligned_all, index + '_optical_' + str(i) + '.png')),
                                        image)
                else:
                    if solver_fail:
                        logging.warning('%s', optical_name)

                # update the progressbar
                pbar.update(1)

        # print statistics
        self.print_counters()
        self.print_averaged_transformation()

    def align_images_mutual_information(self, optical, thermal, thermal_raw, t_init):
        transformation = np.copy(t_init)
        success = True

        # first refine the alignment with the image pyramid if requested
        if self.__params['use_image_pyramid']:
            if not self.__params['perspective']:
                raise ValueError('The image pyramid is only implemented for the perspective transform')

            optical_images = collections.deque([optical])
            thermal_images = collections.deque([thermal.astype(np.float32) / 65535.0])

            # create the image pyramid
            ratio = 1.0
            for i in range(self.__params['alignment/n_pyramid_levels']):
                ratio *= 0.5
                filter_size = int(np.ceil(self.__params['alignment/filter_size'] * ratio))
                if (filter_size % 2) == 0:
                    filter_size += 1
                optical_images.appendleft(cv2.GaussianBlur(optical_images[0],(filter_size,filter_size),0)[::2,::2])
                thermal_images.appendleft(cv2.GaussianBlur(thermal_images[0],(filter_size,filter_size),0)[::2,::2])

            # remove the highest level since we are doing that one later
            optical_images.pop()
            thermal_images.pop()

            for opt_im, th_im in zip(optical_images, thermal_images):
                ratio_x = float(opt_im.shape[0]) / float(optical.shape[0])
                ratio_y = float(opt_im.shape[1]) / float(optical.shape[1])
                transformation[0,1:] *= ratio_x
                transformation[1:,0] /= ratio_x

                transformation[1,0] *= ratio_y
                transformation[1,2] *= ratio_y
                transformation[0,1] /= ratio_y
                transformation[2,1] /= ratio_y

                (success, solver_fail, best_warped, transformation, valid_warped,
                 self.__counter) = align_images(opt_im,
                                                th_im,
                                                transformation,
                                                self.__params,
                                                self.__counter,
                                                self.__params['verbose'],
                                                self.__params['show_results'],
                                                filter_images=False)

                if success:
                    transformation[0,1:] /= ratio_x
                    transformation[1:,0] *= ratio_x

                    transformation[1,0] /= ratio_y
                    transformation[1,2] /= ratio_y
                    transformation[0,1] *= ratio_y
                    transformation[2,1] *= ratio_y
                else:
                    transformation = np.copy(t_init)

        # second refine the alignment with the smoothed image if requested
        if self.__params['use_smoothing_stage']:
            success, solver_fail, _, transformation, _, _ = align_images(optical,
                                                                         thermal.astype(np.float32) / 65535.0,
                                                                         transformation,
                                                                         self.__params,
                                                                         self.__counter,
                                                                         self.__params['verbose'],
                                                                         self.__params['show_results'],
                                                                         filter_images=True)

        if not success:
            # overwrite again t_init and start from there
            transformation = np.copy(t_init)

        # refine the image with the full scale original image
        is_t_init = transformation == t_init
        (success, solver_fail, best_warped, transformation, valid_warped,
         self.__counter) = align_images(optical,
                                        thermal.astype(np.float32) / 65535.0,
                                        transformation,
                                        self.__params,
                                        self.__counter,
                                        self.__params['verbose'],
                                        self.__params['show_results'],
                                        filter_images=False)

        if not success and not is_t_init.all():
            # try again this time with the correct init position
            (success, solver_fail, best_warped, transformation, valid_warped,
             self.__counter) = align_images(optical,
                                            thermal.astype(np.float32) / 65535.0,
                                            t_init,
                                            self.__params,
                                            self.__counter,
                                            self.__params['verbose'],
                                            self.__params['show_results'],
                                                    filter_images=False)

        return success, solver_fail, best_warped, transformation, valid_warped


    def update_average_transformation(self, transformation):
        '''
        Only considers a 3x3 perspective transformation matrix.
        Updates the average rotations and translations by decomposing
        the transformation matrix and then updating the running average
        for each component.

        Parameters
        ----------
        transformation: numpy.array
            Input transformation matrix
        '''

        if transformation.shape == (3,3):
            num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(transformation,
                                                          np.array([[1.0,0,0],[0.0,1.0,0],[0.0,0.0,1.0]]))
            self.__averaged_n_measurements += 1.0
            self.__averaged_rotation += (rotationMatrixToEulerAngles(Rs[2]) -  self.__averaged_rotation) /  self.__averaged_n_measurements
            self.__averaged_translation += (Ts[2] - self.__averaged_translation) / self.__averaged_n_measurements

    def print_averaged_transformation(self):
        '''
        Print the average transformation (euler angles and translation)
        '''

        print('---------')
        print('Rotation: (' +
              str(self.__averaged_rotation[0]) + ', ' +
              str(self.__averaged_rotation[1]) + ', ' +
              str(self.__averaged_rotation[2]) + ')')
        print('Translation: (' +
              str(self.__averaged_translation[0][0]) + ', ' +
              str(self.__averaged_translation[1][0]) + ', ' +
              str(self.__averaged_translation[2][0]) + ')')


    def print_counters(self):
        '''
        Print the counters indicating which method has been used
        to align the images and the counters indicating which
        items have been purged.
        '''

        print('---------')
        print('Alignment method counters:')
        print('  Number of pairs:             ' + str(self.__counter['total']))
        for key in self.__counter.keys():
            if not key == 'total':
                name = key
                if name == '0':
                    name = 'init'
                print('   ' + name + ': ' + str(self.__counter[key]))

def main():
    parser = argparse.ArgumentParser(description='Extract and align the images from a rosbag')
    parser.add_argument('-y', '--yaml-config', default='config_align_images.yaml', help='Yaml file containing the configs')
    parser.add_argument('-i', '--input-dir', default='/tmp/data', help='Input directory')
    parser.add_argument('-o', '--output-dir', default='/tmp/data/processed', help='Output directory')

    args = parser.parse_args()

    worker = ImageAligner(args.input_dir, args.output_dir, args.yaml_config)

    worker.run()

if __name__ == "__main__":
    main()
