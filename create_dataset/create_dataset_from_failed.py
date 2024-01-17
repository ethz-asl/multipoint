import argparse
import os
from pathlib import Path
from shutil import copyfile

def main():
    '''
    TODO
    '''
    parser = argparse.ArgumentParser(description='Show the aligned images as gifs and manually accept/reject pairs')
    parser.add_argument('-f', '--input-file', default='/tmp/data/in/failed.log', help='Input file containing the names of the failed alignments')
    parser.add_argument('-i', '--input-dir', default='/tmp/data/in/preprocessed', help='Input directory containing all the processed image pairs')
    parser.add_argument('-o', '--output-dir', default='/tmp/data/out/preprocessed', help='Output directory')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    failed_names = []
    with open(str(Path(args.input_file))) as fp:
        line = fp.readline()
        while line:
            failed_names.append(line.rstrip())
            line = fp.readline()

    copyfile(str(Path(args.input_dir, 'initial_transform.yaml')), str(Path(args.output_dir, 'initial_transform.yaml')))

    for failed in failed_names:
        index = failed.split('_')[0]
        copyfile(str(Path(args.input_dir, failed)), str(Path(args.output_dir, failed)))
        copyfile(str(Path(args.input_dir, index + '_thermal.png')), str(Path(args.output_dir, index + '_thermal.png')))
        copyfile(str(Path(args.input_dir, index + '_thermal_raw.png')), str(Path(args.output_dir, index + '_thermal_raw.png')))

if __name__ == "__main__":
    main()