import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import yaml

import multipoint.datasets as datasets
import multipoint.models as models
import multipoint.utils as utils

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(description='Predict the keypoints of an image')
    parser.add_argument('-y', '--yaml-config', default='configs/config_image_pair_dataset_prediction.yaml', help='YAML config file')
    parser.add_argument('-m', '--model-dir', default='model_weights/multipoint', help='Directory of the model')
    parser.add_argument('-v', '--version', default='latest', help='Model version (name of the param file), none for no weights')
    parser.add_argument('-i', '--index', default=0, type=int, help='Index of the sample to predict and show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
    parser.add_argument('-p', dest='plot', action='store_true', help='If set the prediction the results are displayed')
    parser.add_argument('-e', dest='evaluation', action='store_true', help='If set the evaluation metrics are computed')
    parser.add_argument('-tk', dest='threshold_keypoints', default=4, type=int, help='Distance below which two keypoints are considered a match')
    parser.add_argument('-th', dest='threshold_homography', default=1, type=int, help='Homography correctness threshold')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        # overwrite the model params
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']

    # check training device
    device = torch.device("cpu")
    if config['prediction']['allow_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Predicting on device: {}'.format(device))

    # dataset
    dataset = getattr(datasets, config['dataset']['type'])(config['dataset'])
    loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=config['prediction']['batchsize'],
                                                 shuffle=False, num_workers=config['prediction']['num_worker'])

    # network
    net = getattr(models, config['model']['type'])(config['model'])
    if args.version != 'none':
        weights = torch.load(os.path.join(args.model_dir, args.version + '.model'), map_location=torch.device('cpu'))
        weights = utils.fix_model_weigth_keys(weights)
        net.load_state_dict(weights)
        del weights
    net.to(device)

    # put the network into the evaluation mode
    net.eval()

    with torch.no_grad():
        if args.evaluation:
            results = utils.compute_descriptor_metrics(net, loader_dataset, device, config['prediction'], args.threshold_keypoints, args.threshold_homography)

            print('NN-mAP: {}'.format(results['nn_map']))
            print('M-Score: {}'.format(results['m_score']))
            print('Homography Correctness: {}'.format(results['h_correctness']))

            # also add the params to store them
            results['config'] = config
            results['threshold_keypoints'] = args.threshold_keypoints
            results['threshold_homography'] = args.threshold_homography

            # save results
            target_dir = os.path.join(args.model_dir, 'descriptor_evaluation')
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            np.save(os.path.join(target_dir, os.path.split(args.model_dir.strip("/"))[-1] + '_' +
                                  time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())), results)

            if args.plot:
                plt.figure()
                plt.title('PR curve')
                plt.xlabel('precision')
                plt.ylabel('recall')
                plt.plot(results['recall_optical'], results['precision_optical'], 'r')
                plt.plot(results['recall_thermal'], results['precision_thermal'], 'g')
                plt.legend(['optical', 'thermal'])

                plt.figure()
                plt.title('Optical M-score')
                plt.hist(results['m_score_optical'], 50)

                plt.figure()
                plt.title('Thermal M-score')
                plt.hist(results['m_score_thermal'], 50)

                plt.figure()
                plt.title('Warp point distance error')
                plt.hist(results['pts_dist'], 50)

                plt.show()

        # get the sample and move it to the right device
        synchronize()
        t_start = time.time()
        data = dataset[args.index]
        data = utils.data_to_device(data, device)
        data = utils.data_unsqueeze(data, 0)

        synchronize()
        t_1 = time.time()
        # predict
        out_optical = net(data['optical'])
        out_thermal = net(data['thermal']) # could be optimized to move into one forward pass
        synchronize()
        t_2 = time.time()

        # compute the nms probablity
        if config['prediction']['nms'] > 0:
            out_optical['prob'] = utils.box_nms(out_optical['prob'] * data['optical']['valid_mask'],
                                                config['prediction']['nms'],
                                                config['prediction']['detection_threshold'],
                                                keep_top_k=config['prediction']['topk'],
                                                on_cpu=config['prediction']['cpu_nms'])
            out_thermal['prob'] = utils.box_nms(out_thermal['prob'] * data['thermal']['valid_mask'],
                                                config['prediction']['nms'],
                                                config['prediction']['detection_threshold'],
                                                keep_top_k=config['prediction']['topk'],
                                                on_cpu=config['prediction']['cpu_nms'])

        synchronize()
        t_3 = time.time()
        print('Loading the data took: {} s'.format(t_1 - t_start))
        print('Two forward passes took: {} s'.format(t_2 - t_1))
        print('Box nms: {} s'.format(t_3 - t_2))

        # display a sample
        if args.plot:
            # add homography to data if not available
            if 'homography' not in data['optical'].keys():
                data['optical']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

            if 'homography' not in data['thermal'].keys():
                data['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

            for i, (optical, thermal,
                    prob_optical, prob_thermal,
                    mask_optical, mask_thermal,
                    H_optical, H_thermal,
                    desc_optical, desc_thermal) in enumerate(zip(data['optical']['image'],
                                                                 data['thermal']['image'],
                                                                 out_optical['prob'],
                                                                 out_thermal['prob'],
                                                                 data['optical']['valid_mask'],
                                                                 data['thermal']['valid_mask'],
                                                                 data['optical']['homography'],
                                                                 data['thermal']['homography'],
                                                                 out_optical['desc'],
                                                                 out_thermal['desc'],)):

                # get the keypoints
                pred_optical = torch.nonzero((prob_optical.squeeze() > config['prediction']['detection_threshold']).float())
                pred_thermal = torch.nonzero((prob_thermal.squeeze() > config['prediction']['detection_threshold']).float())
                kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.cpu().numpy().astype(np.float32)]
                kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.cpu().numpy().astype(np.float32)]

                # get the descriptors
                if desc_optical.shape[1:] == prob_optical.shape[1:]:
                    # classic descriptors, directly take values
                    desc_optical_sampled = desc_optical[:, pred_optical[:,0], pred_optical[:,1]].transpose(0,1)
                    desc_thermal_sampled = desc_thermal[:, pred_thermal[:,0], pred_thermal[:,1]].transpose(0,1)
                else:
                    H, W = data['optical']['image'].shape[2:]
                    desc_optical_sampled = utils.interpolate_descriptors(pred_optical, desc_optical, H, W)
                    desc_thermal_sampled = utils.interpolate_descriptors(pred_thermal, desc_thermal, H, W)

                # match the keypoints
                matches = utils.get_matches(desc_optical_sampled.cpu().numpy(),
                                            desc_thermal_sampled.cpu().numpy(),
                                            config['prediction']['matching']['method'],
                                            config['prediction']['matching']['knn_matches'],
                                            **config['prediction']['matching']['method_kwargs'])

                # mask the image if requested
                optical *= mask_optical
                thermal *= mask_thermal 

                # convert images to numpy arrays
                im_optical = cv2.cvtColor((np.clip(optical.squeeze().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
                im_thermal = cv2.cvtColor((np.clip(thermal.squeeze().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)



                # draw the matches
                out_image = cv2.drawMatches(im_optical, kp_optical, im_thermal, kp_thermal, matches, None, flags=2)
                cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('matches', (out_image.shape[1]*2, out_image.shape[0]*2 + 50))
                cv2.imshow('matches', out_image)

                # align images to estimate homography and get good matches
                optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

                if optical_pts.shape[0] < 4 or thermal_pts.shape[0] < 4:
                    H_est = np.eye(3,3)
                    matchesMask = []
                else:
                    H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=config['prediction']['reprojection_threshold'])
                    matchesMask = mask.ravel().tolist()

                warped_image = cv2.warpPerspective(im_optical, H_est, im_optical.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT)
                cv2.imshow('warped optical', warped_image)


                # correct matches mask
                H_gt = np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy()))
                warped_optical = utils.warp_keypoints(optical_pts.squeeze()[:,::-1], H_gt)[:,::-1]
                diff = thermal_pts.squeeze() - warped_optical
                diff = np.linalg.norm(diff, axis=1)
                matchesMask = (diff < 4.0).tolist()

                # draw refined matches
                out_image_refined = cv2.drawMatches(im_optical,
                                                    kp_optical,
                                                    im_thermal,
                                                    kp_thermal,
                                                    matches, 
                                                    None,
                                                    matchColor=(0, 255, 0),
                                                    singlePointColor=(0, 0, 255),
                                                    flags=0,
                                                    matchesMask = matchesMask)

                cv2.namedWindow('matches_refined', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('matches_refined', (out_image_refined.shape[1]*2, out_image_refined.shape[0]*2 + 50))
                cv2.imshow('matches_refined', out_image_refined)

                # compare estimated and computed homography
                print('--------------------------------------------------------')
                print('Estimated Homography:')
                print(H_est)
                print('Ground Truth Homography:')
                print(np.matmul(H_thermal.cpu().numpy(), np.linalg.inv(H_optical.cpu().numpy())))
                print('--------------------------------------------------------')

            cv2.waitKey(0)

if __name__ == "__main__":
    main()
