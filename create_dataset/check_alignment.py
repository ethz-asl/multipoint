import argparse
import cv2
import logging
import os
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Show the aligned images as gifs and manually accept/reject pairs')
    parser.add_argument('-i', '--input-dir', default='/tmp/data', help='Input directory')
    parser.add_argument('-dt', '--dt', type=int, default=500, help='Time to show images in ms')
    parser.add_argument('-a', '--accept', default='a', help='Alternative accept key')
    parser.add_argument('-r', '--reject', default='r', help='Alternative reject key')


    args = parser.parse_args()

    best_path = str(Path(args.input_dir, 'aligned', 'best'))
    all_path = str(Path(args.input_dir, 'aligned', 'all'))
    accepted_path = str(Path(args.input_dir, 'aligned', 'accepted'))

    if not os.path.isdir(accepted_path):
        os.makedirs(accepted_path)

    optical_filenames = [f for f in os.listdir(best_path)
                         if os.path.isfile(os.path.join(best_path, f)) and 'optical' in f]

    print('----------------------------------------------------------')
    print('Instructions:')
    print('First the images from the best alignment will be shown.')
    print('This alignment can be accepted by pressing the "a" key')
    print('or rejected using the "r" key.')
    print('If the alignment is accepted the next pair is shown.')
    print('If it is rejected all the valid pairs resulting from')
    print('a valid alignment will be shown except for the case')
    print('where the best alignment was the only valid one.')
    print('By pressing the key with the number of the title of the')
    print('GIF the respecive pair can be accepted. If no pair is')
    print('acceptable all the pairs can be rejected by the "n" key.')
    print('----------------------------------------------------------')

    if args.accept != 'a':
        print('Alternative accept key selected: {0}'.format(args.accept))
    if args.reject != 'r':
        print('Alternative reject key selected: {0}'.format(args.reject))

    logging.basicConfig(filename=str(Path(args.input_dir, 'checked.log')),
                        level=logging.DEBUG,
                        format='%(message)s')

    # remove the already checked files from the list
    already_checked = []
    with open(str(Path(args.input_dir, 'checked.log'))) as fp:
        line = fp.readline()
        while line:
            if not line == '':
                already_checked.append(line.rstrip())
            line = fp.readline()

    for item in already_checked:
        optical_filenames.remove(item)

    accepted_counter = 0
    for optical_name in tqdm(optical_filenames):
        # load best aligned images
        index = optical_name.split('_')[0]
        optical_path = str(Path(best_path, optical_name))
        thermal_path = str(Path(best_path, index + '_thermal.png'), )
        thermal_raw_path = str(Path(best_path, index + '_thermal_raw.png'), )

        # load the images
        optical = cv2.imread(str(optical_path), -1)
        thermal = cv2.imread(str(thermal_path), -1)
        thermal_raw = cv2.imread(str(thermal_raw_path), -1)

        # start with not accepted
        accepted = False
        accepted_optical = None

        # show the best aligned images until it is accepted or rejected
        image = optical
        is_1= True

        while True:
            cv2.imshow('best aligned', image)
            cv2.namedWindow('best aligned',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('best aligned', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            k = cv2.waitKey(args.dt)
            if k > -1:
                k = chr(k)
                if k == 'a' or k == args.accept:
                    accepted = True
                    accepted_optical = optical
                    break
                elif k == 'r' or k == args.reject:
                    accepted = False
                    break
            if is_1:
                is_1 = False
                image = thermal
            else:
                is_1 = True
                image = optical

        cv2.imshow('best aligned', 0.0*optical)

        if not accepted:
            # get the alternative names
            optical_alternative_filenames = [f for f in os.listdir(all_path)
                                             if os.path.isfile(os.path.join(all_path, f)) and f.startswith(index + '_optical')]

            num_images = len(optical_alternative_filenames)

            # load the optical images
            optical_all = []
            for name in optical_alternative_filenames:
                optical_all.append(cv2.imread(str(str(Path(all_path, name))), -1))

            thermal_all = len(optical_all) * [thermal]

            images = optical_all
            is_1= True

            while True:
                for i, im in enumerate(images):
                    cv2.imshow(str(i), im)
                    cv2.namedWindow(str(i),cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(str(i), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                k = cv2.waitKey(args.dt)
                if k > -1:
                    k = chr(k)
                    if k == 'n':
                        accepted = False
                        break
                    elif k == 'b':
                        accepted = True
                        accepted_optical = optical
                        break
                    elif k.isdigit():
                        k = int(k)
                        if (k>=0) and (k < len(optical_all)):
                            accepted = True
                            accepted_optical = optical_all[k]
                            break
                if is_1:
                    is_1 = False
                    images = thermal_all
                else:
                    is_1 = True
                    images = optical_all

            for i in range(num_images):
                cv2.imshow(str(i), 0.0*optical)

        # save images if accepted
        if accepted:
            optical_path_accepted = str(Path(accepted_path, optical_name))
            thermal_path_accepted = str(Path(accepted_path, index + '_thermal.png'), )
            thermal_raw_path_accepted = str(Path(accepted_path, index + '_thermal_raw.png'), )

            cv2.imwrite(optical_path_accepted, accepted_optical)
            cv2.imwrite(thermal_path_accepted, thermal)
            cv2.imwrite(thermal_raw_path_accepted, thermal_raw)
            accepted_counter += 1

        # mark this sample as checked
        logging.warning('%s', optical_name)

    print('Accepted ', accepted_counter, ' images out of ', len(optical_filenames))

if __name__ == "__main__":
    main()
