import argparse
import copy
import os
from pathlib import Path
from shutil import copyfile

def main():
    '''
    If the alignment process was run twice try to find the failed images from the output directory
    in the input directory and copy them to the output directory.
    '''
    parser = argparse.ArgumentParser(description='Show the aligned images as gifs and manually accept/reject pairs')
    parser.add_argument('-i', '--input-dir', default='/tmp/data/in', help='Input directory')
    parser.add_argument('-o', '--output-dir', default='/tmp/data/out', help='Output directory')

    args = parser.parse_args()

    in_path = str(Path(args.input_dir, 'aligned'))
    out_path = str(Path(args.output_dir, 'aligned'))

    failed_names = []
    with open(str(Path(args.output_dir, 'failed.log'))) as fp:
        line = fp.readline()
        while line:
            failed_names.append(line.rstrip())
            line = fp.readline()

    input_names = [f for f in os.listdir(str(Path(in_path, 'best')))
                         if os.path.isfile(os.path.join(str(Path(in_path, 'best')), f)) and 'optical' in f]

    not_added = copy.deepcopy(failed_names)
    added_indices = []
    for failed in failed_names:
        for input in input_names:
            if failed == input:
                index = input.split('_')[0]
                copyfile(str(Path(in_path, 'best', input)), str(Path(out_path, 'best', input)))
                copyfile(str(Path(in_path, 'best', index + '_thermal.png')), str(Path(out_path, 'best', index + '_thermal.png')))
                copyfile(str(Path(in_path, 'best', index + '_thermal_raw.png')), str(Path(out_path, 'best', index + '_thermal_raw.png')))

                optical_all_filenames = [f for f in os.listdir(str(Path(in_path, 'all')))
                                         if os.path.isfile(os.path.join(str(Path(in_path, 'all')), f)) and f.startswith(index + '_optical')]

                for name in optical_all_filenames:
                    copyfile(str(Path(in_path, 'all', name)), str(Path(out_path, 'all', name)))

                not_added.remove(failed)
                added_indices.append(index)

    with open(str(Path(args.output_dir, 'failed.log')), "w") as fp:
        for line in not_added:
            fp.write(line + '\n')

    print('Added images with the following indices:')
    print(added_indices)

if __name__ == "__main__":
    main()