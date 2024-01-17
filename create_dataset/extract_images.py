import argparse
import cv2
import cv_bridge
import math
import numpy as np
import os
from pathlib import Path
import rosbag
import rospy
from tqdm import tqdm
import yaml

from helper_functions.disp import *
from helper_functions.utils import *

class ImageExtractorRos():
    '''
    Class to iterate over a bagfile and find image pairs based on the timestamp.
    '''

    def __init__(self, input_dir, output_dir, bagfile, config_file):
        '''
        Initializer
        
            Parameters
        ----------
        input_dir : string
            Directory containing the bagfiles and the calibration file
        output_dir : string
            Directory where the processed images will be stored
        bagfile : string
            Name of the bagfile to process
        config_file : string
            Param file
        '''

        self.__input_dir = input_dir
        self.__output_dir = output_dir
        self.__bagfile = bagfile

        with open(config_file, 'rt') as fh:
            self.__params = yaml.safe_load(fh)

        if self.__params['undistort_images']:
            calibration_file_path = os.path.join(self.__input_dir, self.__params['image/calibration_params'])
            with open(calibration_file_path, 'rt') as fh:
                self.__calibration_params = yaml.safe_load(fh)

        bag_path = os.path.join(self.__input_dir, self.__bagfile)
        self.__bag = rosbag.Bag(bag_path)
        self.__num_thermal_images = self.__bag.get_message_count(self.__params['rosbag/topic_thermal_images'])
        self.__num_optical_images = self.__bag.get_message_count(self.__params['rosbag/topic_optical_images'])
        print('Number of thermal images: ' + str(self.__num_thermal_images))
        print('Number of optical images: ' + str(self.__num_optical_images))

        self.__queue_optical_images_raw = []
        self.__queue_thermal_images_raw = []
        self.__queue_pose = []
        self.__queue_exposure_times = []
        self.__queue_optical_images_compensated = []
        self.__queue_thermal_images_compensated = []
        self.__queue_optical_images_processed = []
        self.__queue_thermal_images_processed = []

        self.__purge_counter = dict()

        self.__cv_bridge = cv_bridge.CvBridge()

    def run(self):
        '''
        Run the extraction process by iterating through the bagfile.

        The following steps are executed:
            - Preprocess the images (undistort, rotate, and downscale)
            - If specified compensate the exposure time
            - If specified reject images based on pose
            - Find image pairs based on timestamp and store the images
        '''

        # create directories if required
        if self.__params['save_preprocessed_images']:
            out_path_preprocessed = Path(self.__output_dir, 'preprocessed')

            if not os.path.isdir(str(out_path_preprocessed)):
                os.makedirs(str(out_path_preprocessed))

        # set the start and end time
        start_time = rospy.Time(self.__bag.get_start_time() + self.__params['rosbag/start_time'])
        end_time = rospy.Time(self.__bag.get_start_time() + self.__params['rosbag/end_time'])
        end_time = min(end_time, rospy.Time(self.__bag.get_end_time()))
        start_time = max(start_time, rospy.Time(self.__bag.get_start_time()))

        # downscale the expected number of iteration according to the requested timespan
        # this assumes that images are uniformly distributed throughout the bag
        num_iter = self.__num_thermal_images
        factor = (end_time - start_time) / (rospy.Time(self.__bag.get_end_time()) - rospy.Time(self.__bag.get_start_time()))
        num_iter = int(num_iter*factor)

        # iterate through bagfile
        pair_counter = 0
        with tqdm(total=num_iter) as pbar:
            for topic, msg, t in self.__bag.read_messages(topics=[self.__params['rosbag/topic_optical_images'],
                                                                  self.__params['rosbag/topic_thermal_images'],
                                                                  self.__params['rosbag/topic_optical_exposure'],
                                                                  self.__params['rosbag/topic_pose']],
                                                          start_time=start_time,
                                                          end_time=end_time):

                # show queue length
                pbar.set_description('O_R: {}, I_R: {}, O_C: {}, I_C: {}, O_P: {}, I_P: {}, E: {}, P: {}'.format(
                    len(self.__queue_optical_images_raw),
                    len(self.__queue_thermal_images_raw),
                    len(self.__queue_optical_images_compensated),
                    len(self.__queue_thermal_images_compensated),
                    len(self.__queue_optical_images_processed),
                    len(self.__queue_thermal_images_processed),
                    len(self.__queue_exposure_times),
                    len(self.__queue_pose),
                    ))

                # check if we want to process the message
                if (topic == self.__params['rosbag/topic_optical_images']):
                    self.__queue_optical_images_raw.append(msg)
                elif (topic == self.__params['rosbag/topic_thermal_images']):
                    self.__queue_thermal_images_raw.append(msg)
                    pbar.update(1)
                elif (topic == self.__params['rosbag/topic_optical_exposure']):
                    if self.__params['compensate_exposure_time']:
                        self.__queue_exposure_times.append(msg)
                elif (topic == self.__params['rosbag/topic_pose']):
                    if self.__params['check_pose']:
                        self.__queue_pose.append(msg)

                # compensate the exposure time if requested (move images from _raw queue to _compensated queue)
                self.compensate_exposure_time()

                # filter images by pose if requested (_compensated queue to _processed queue)
                self.filter_images_by_pose()

                # purge old messages from queues
                self.purge_old_messages(msg)

                # if there are images in the queue check for pairs
                if (len(self.__queue_optical_images_processed) > 0 and len(self.__queue_thermal_images_processed) > 0):
                    found_pair, optical_msg, thermal_msg = self.get_image_pair()
                    if found_pair:
                        optical_image, thermal_image, thermal_image_rescaled = self.preprocess_images(optical_msg, thermal_msg)

                        if self.__params['save_preprocessed_images']:
                            cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_optical.png')),
                                        optical_image)
                            cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_thermal_raw.png')),
                                        thermal_image)
                            cv2.imwrite(str(Path(out_path_preprocessed, str(pair_counter) + '_thermal.png')),
                                        (thermal_image_rescaled * 65535).astype('uint16'))

                        if self.__params['image/show_raw/dt'] > 0:
                            show_image_pair(optical_image, thermal_image_rescaled, self.__params['image/show_raw/dt'])

                        pair_counter += 1

        # print statistics
        self.print_counters()

        self.__bag.close()

    def preprocess_images(self, optical_msg, thermal_msg):
        '''
        Preprocess the images according to the settings.
        - First convert them from ros messages to opencv images
        - If requested undistort the images
        - If requested rotate the thermal image by 180 degrees
        - If requested downscale the optical image
        - Normalize the thermal image (with outlier rejection if requested)

        Parameters
        ----------
        optical_msg: sensor_msgs/Image
            ROS message of the optical image
        thermal_msg: sensor_msgs/Image
            ROS message of the thermal image

        Returns
        -------
        cv_optical : numpy.array
            Preprocessed optical image in the opencv format
        cv_thermal : numpy.array
            Preprocessed optical image in the opencv format
            with the original raw pixel values [uint16]
        cv_thermal_rescaled : numpy.array
            Preprocessed thermal image in the opencv format.
            The pixel values are rescaled between 0 to 1 [float]
        '''

        # extract the images from the messages
        cv_optical = self.__cv_bridge.imgmsg_to_cv2(optical_msg, "bgr8")
        cv_thermal = self.__cv_bridge.imgmsg_to_cv2(thermal_msg, "mono16")

        # undistort the images if requested
        if self.__params['undistort_images']:
            for params in self.__calibration_params['cameras']:
                camera_matrix = get_camera_matrix(params['camera']['intrinsics']['data'])
                distortion_params = np.array(params['camera']['distortion']['parameters']['data'])

                if params['camera']['label'] == 'optical':
                    h, w = cv_optical.shape[:2]
                    camera_matrix_opt, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                                           distortion_params,
                                                                           (w, h),
                                                                           self.__params['image/undistort_alpha'])
                    cv_optical = cv2.undistort(cv_optical, camera_matrix, distortion_params,  None, camera_matrix_opt)
                elif params['camera']['label'] == 'thermal':
                    h, w = cv_thermal.shape[:2]
                    camera_matrix_opt, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                                           distortion_params,
                                                                           (w, h),
                                                                           self.__params['image/undistort_alpha'])
                    cv_thermal = cv2.undistort(cv_thermal, camera_matrix, distortion_params, None, camera_matrix_opt)
                else:
                    raise ValueError('ERROR unknown camera label: ' +  params['camera']['label'])

        # rotate the thermal image if requested
        if self.__params['image/thermal/rotate']:
            cv_thermal = cv_thermal[...,::-1,::-1]

        # downscale the optical if desired
        if self.__params['image/optical/downscale']:
            ratio = float(cv_thermal.shape[0])/cv_optical.shape[0]
            cv_optical = cv2.resize(cv_optical,(int(cv_optical.shape[1]*ratio), int(cv_thermal.shape[0])))

        # convert thermal image to a floating point image and rescale it
        if self.__params['image/thermal/rescale_outlier_rejection']:
            lower_bound = np.percentile(cv_thermal, 1)
            upper_bound = np.percentile(cv_thermal, 99)

            cv_thermal_rescaled = cv_thermal
            cv_thermal_rescaled[cv_thermal_rescaled < lower_bound] = lower_bound
            cv_thermal_rescaled[cv_thermal_rescaled > upper_bound] = upper_bound

        cv_thermal_rescaled = cv2.normalize(cv_thermal_rescaled, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return cv_optical, cv_thermal, cv_thermal_rescaled

    def compensate_exposure_time(self):
        '''
        Compensate the timestamp of the optical images by the exposure time.
        Moves the images from the optical_images_raw queue to the
        optical_images_compensated queue.
        '''

        if self.__params['compensate_exposure_time']:
            if len(self.__queue_optical_images_raw) > 0 and len(self.__queue_exposure_times):
                for exposure in self.__queue_exposure_times:
                    for image in self.__queue_optical_images_raw:
                        if image.header.stamp == exposure.header.stamp:
                            image.header.stamp -= exposure.exposure
                            self.__queue_optical_images_compensated.append(image)
                            self.__queue_optical_images_raw.remove(image)
                            self.__queue_exposure_times.remove(exposure)
        else:
            # copy images to next queue
            self.__queue_optical_images_compensated.extend(self.__queue_optical_images_raw)
            self.__queue_optical_images_raw = []
            self.__queue_exposure_times = []

        # nothing to do for thermal images, just copy them
        self.__queue_thermal_images_compensated.extend(self.__queue_thermal_images_raw)
        self.__queue_thermal_images_raw = []

    def filter_images_by_pose(self):
        '''
        Check the pose at the time the image was captures.
        If the pose is within the specified bounds (roll and pitch)
        move the images from the *_compensated queue to the
        *_processed queue, otherwise leave the images in the *_compensated.
        '''

        if self.__params['check_pose']:
            # optical image
            for image in self.__queue_optical_images_compensated:
                for pose in self.__queue_pose:
                    dt = abs(get_timestamp(image) - get_timestamp(pose))
                    if dt < self.__params['rosbag/check_pose/max_dt']:
                        # convert to rpy
                        roll, pitch, yaw = quat2euler(pose.pose.orientation.x,
                                                      pose.pose.orientation.y,
                                                      pose.pose.orientation.z,
                                                      pose.pose.orientation.w)
                        roll = math.degrees(roll)
                        pitch = math.degrees(pitch) - self.__params['rosbag/check_pose/pitch_offset']
                        yaw = math.degrees(yaw)

                        # check if pose is within the bounds
                        if ((abs(roll) < self.__params['rosbag/check_pose/max_roll']) and
                            (abs(pitch) < self.__params['rosbag/check_pose/max_pitch'])):
                            self.__queue_optical_images_processed.append(image)
                            self.__queue_optical_images_compensated.remove(image)
                            break

            for image in self.__queue_thermal_images_compensated:
                for pose in self.__queue_pose:
                    dt = abs(get_timestamp(image) - get_timestamp(pose))
                    if dt < self.__params['rosbag/check_pose/max_dt']:
                        # convert to rpy
                        roll, pitch, yaw = quat2euler(pose.pose.orientation.x,
                                                      pose.pose.orientation.y,
                                                      pose.pose.orientation.z,
                                                      pose.pose.orientation.w)
                        roll = math.degrees(roll)
                        pitch = math.degrees(pitch)
                        yaw = math.degrees(yaw)

                        # check if pose is within the bounds
                        if ((abs(roll) < self.__params['rosbag/check_pose/max_roll']) and
                            (abs(pitch) < self.__params['rosbag/check_pose/max_pitch'])):
                            self.__queue_thermal_images_processed.append(image)
                            self.__queue_thermal_images_compensated.remove(image)
                            break

        else:
            self.__queue_thermal_images_processed.extend(self.__queue_thermal_images_compensated)
            self.__queue_optical_images_processed.extend(self.__queue_optical_images_compensated)
            self.__queue_thermal_images_compensated = []
            self.__queue_optical_images_compensated = []

    def purge_old_messages(self, current_msg):
        '''
        Remove messages from the queues which are behind the timestamp
        of the current message specified by the 'rosbag/purge_dt'
        parameter.

        Parameters
        ----------
        current_msg: ros.message
            Arbitrary ros message with a header
        '''

        current_timestamp = get_timestamp(current_msg) - self.__params['rosbag/purge_dt']
        for item in self.__queue_exposure_times:
            if get_timestamp(item) < current_timestamp:
                self.__queue_exposure_times.remove(item)
                if 'exposure_times' in self.__purge_counter:
                    self.__purge_counter['exposure_times'] += 1
                else:
                    self.__purge_counter['exposure_times'] = 1

        for item in self.__queue_pose:
            if get_timestamp(item) < current_timestamp:
                self.__queue_pose.remove(item)
                if 'pose' in self.__purge_counter:
                    self.__purge_counter['pose'] += 1
                else:
                    self.__purge_counter['pose'] = 1

        for item in self.__queue_optical_images_compensated:
            if get_timestamp(item) < current_timestamp:
                self.__queue_optical_images_compensated.remove(item)
                if 'optical_compensated' in self.__purge_counter:
                    self.__purge_counter['optical_compensated'] += 1
                else:
                    self.__purge_counter['optical_compensated'] = 1

        for item in self.__queue_optical_images_processed:
            if get_timestamp(item) < current_timestamp:
                self.__queue_optical_images_processed.remove(item)
                if 'optical_processed' in self.__purge_counter:
                    self.__purge_counter['optical_processed'] += 1
                else:
                    self.__purge_counter['optical_processed'] = 1

        for item in self.__queue_optical_images_raw:
            if get_timestamp(item) < current_timestamp:
                self.__queue_optical_images_raw.remove(item)
                if 'optical_raw' in self.__purge_counter:
                    self.__purge_counter['optical_raw'] += 1
                else:
                    self.__purge_counter['optical_raw'] = 1

        for item in self.__queue_thermal_images_compensated:
            if get_timestamp(item) < current_timestamp:
                self.__queue_thermal_images_compensated.remove(item)
                if 'thermal_compensated' in self.__purge_counter:
                    self.__purge_counter['thermal_compensated'] += 1
                else:
                    self.__purge_counter['thermal_compensated'] = 1

        for item in self.__queue_thermal_images_processed:
            if get_timestamp(item) < current_timestamp:
                self.__queue_thermal_images_processed.remove(item)
                if 'thermal_processed' in self.__purge_counter:
                    self.__purge_counter['thermal_processed'] += 1
                else:
                    self.__purge_counter['thermal_processed'] = 1

        for item in self.__queue_thermal_images_raw:
            if get_timestamp(item) < current_timestamp:
                self.__queue_thermal_images_raw.remove(item)
                if 'thermal_raw' in self.__purge_counter:
                    self.__purge_counter['thermal_raw'] += 1
                else:
                    self.__purge_counter['thermal_raw'] = 1

    def get_image_pair(self):
        '''
        Iterate through the optical and thermal image queue to find an image pair.
        An image pair is found if the compensated timestamps are within the specification.
        The pair is returned and the images are removed from the queue.

        Returns
        -------
        found_pair : bool
            Indicates if an image pair was found
        optical: sensor_msgs/Image
            ROS message of the optical image
        thermal: sensor_msgs/Image
            ROS message of the thermal image
        '''

        for thermal in self.__queue_thermal_images_processed:
            for optical in self.__queue_optical_images_processed:
                dt = get_timestamp(thermal) + self.__params['rosgab/thermal_dt'] - get_timestamp(optical)
                if (abs(dt) < self.__params['rosbag/max_dt']):
                    self.__params['rosgab/thermal_dt'] -= dt * self.__params['rosgab/filter_alpha']
                    if self.__params['verbose']:
                        print('Found image pair with dt = ' + str(dt) + ' [ms]')
                    # remove the images from the queue
                    self.__queue_thermal_images_processed.remove(thermal)
                    self.__queue_optical_images_processed.remove(optical)
                    return True, optical, thermal

        return False, None, None

    def print_counters(self):
        '''
        Print the counters indicating which items have been purged.
        '''

        print('---------')
        print('Purge counter:')
        for key in self.__purge_counter.keys():
            print(' ' + key + ': ' + str(self.__purge_counter[key]))

def main():
    parser = argparse.ArgumentParser(description='Extract images from a rosbag and save them as pairs')
    parser.add_argument('-y', '--yaml-config', default='config_extract_images.yaml', help='Yaml file containing the configs')
    parser.add_argument('-i', '--input-dir', default='/tmp/data', help='Input directory')
    parser.add_argument('-b', '--bagfile', default='input.bag', help='Input rosbag file name')
    parser.add_argument('-o', '--output-dir', default='/tmp/data/processed', help='Output directory')

    args = parser.parse_args()

    worker = ImageExtractorRos(args.input_dir, args.output_dir, args.bagfile, args.yaml_config)

    worker.run()

if __name__ == "__main__":
    main()
