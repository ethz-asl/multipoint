import os
import argparse
from utils.data_downloader import MrDataGrabber


def download_multipoint_data():
    base_url = 'http://robotics.ethz.ch/~asl-datasets/2020_ALTAIR_multispectral_dataset/'
    parser = argparse.ArgumentParser(description='Download multipoint dataset')
    parser.add_argument('--test-url', default=base_url+'test.hdf5', help='Test data URL')
    parser.add_argument('--train-url', default=base_url + 'training.hdf5', help='Training data URL')
    parser.add_argument('--labels-url', default=base_url + 'labels_training.hdf5', help='Label training data URL')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Force overwrite existing files')
    parser.add_argument('-d', '--target-dir', default='data', help='Target data directory')
    args = parser.parse_args()

    if args.target_dir is not 'data':
        # If downloading to another location
        print('Data will be downloaded to: {0}'.format(args.target_dir))
        print('Please modify your configuration scripts to use this new location, or symlink your data')

    # Download data
    files = []
    for url in [args.test_url, args.train_url, args.labels_url]:
        downloader = MrDataGrabber(url, args.target_dir, overwrite=args.force_overwrite)
        downloader.download()
        files.append(downloader.target_file)

    return files


if __name__ == "__main__":
    download_multipoint_data()
