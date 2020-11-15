import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import yaml

import multipoint.datasets as datasets
import multipoint.models as models
import multipoint.utils as utils

def main():
    parser = argparse.ArgumentParser(description='Predict the keypoints of an image')
    parser.add_argument('-y', '--yaml-config', default='configs/config_image_pair_dataset_prediction.yaml', help='YAML config file')
    parser.add_argument('-m', '--model-dir', default='model_weights/multipoint', help='Directory of the model')
    parser.add_argument('-v', '--version', default='latest', help='Model version (name of the param file)')
    parser.add_argument('-i', '--index', default=0, type=int, help='Index of the sample to predict and show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
    parser.add_argument('-p', dest='plot', action='store_true', help='If set the prediction the results are displayed')
    parser.add_argument('-e', dest='evaluation', action='store_true', help='If set the evaluation metrics are computed')
    parser.add_argument('-b', dest='batch', action='store_true', help='If set a batch of images is predicted and displayed instead of a single image')
    parser.add_argument('-t', dest='threshold', default=3, type=int, help='Distance threshold for two keypoints to be considered a match')
    parser.add_argument('-mask', dest='mask', action='store_true', help='If set invalid image pixels will be set to 0')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')

    args = parser.parse_args()

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
    net.to(device)

    # put the network into the evaluation mode
    net.eval()

    with torch.no_grad():
        # compute the performance metrics
        if args.evaluation:
            # set the random seed and make the prediction deterministic
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            target_dir = os.path.join(args.model_dir, 'detector_evaluation')
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            # if we have an image pair we can compute the repeatability, else compute the single image metrics
            if dataset.returns_pair():
                repeatability_mean, repeatability, n_kp_optical, n_kp_thermal = utils.compute_repeatability_multispectral(
                    net, loader_dataset, device, config, distance_thresh=args.threshold)
                print('Repeatability: {}'.format(repeatability_mean))
                print('Number of optical keypoints: {}'.format(np.mean(n_kp_optical)))
                print('Number of thermal keypoints: {}'.format(np.mean(n_kp_thermal)))

                # combine results
                results = {}
                results['repeatability_mean'] = repeatability_mean
                results['repeatability'] = repeatability
                results['n_kp_optical'] = n_kp_optical
                results['n_kp_thermal'] = n_kp_thermal
                results['distance_threshold'] = args.threshold
                results['config'] = config

            else:
                precision, recall, prob, dist = utils.compute_detector_metrics(net,
                                                                               loader_dataset,
                                                                               device,
                                                                               config['prediction'])

                # combine results
                results = {}
                results['precision'] = precision
                results['recall'] = recall
                results['prob'] = prob
                results['dist'] = dist
                results['config'] = config

                print('Average distance error for true positives: {}'.format(dist.mean()))
                print('mAP: {}'.format(utils.compute_mAP(precision, recall)))

                if args.plot:
                    plt.plot(recall, precision)
                    plt.show()

            # save results
            np.save(os.path.join(target_dir,
                                 os.path.split(args.model_dir.strip("/"))[-1] + '_' +
                                 time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '.npz'), results)

        # get the sample and move it to the right device
        t_start = time.time()
        if args.batch:
            for i in range(args.index + 1):
                data = next(iter(loader_dataset))
        else:
            data = dataset[args.index]

        data = utils.data_to_device(data, device)

        if not args.batch:
            data = utils.data_unsqueeze(data, 0)

        # predict
        if dataset.returns_pair():
            batch_size = data['optical']['image'].shape[0]
            out_optical = net(data['optical'])
            out_thermal = net(data['thermal'])

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                out_optical['prob'] = utils.box_nms(out_optical['prob'],
                                                    config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])
                out_thermal['prob'] = utils.box_nms(out_thermal['prob'],
                                                    config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])
        else:
            batch_size = data['image'].shape[0]
            out = net(data)

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                out['prob'] = utils.box_nms(out['prob'],
                                            config['prediction']['nms'],
                                            config['prediction']['detection_threshold'],
                                            keep_top_k=config['prediction']['topk'],
                                            on_cpu=config['prediction']['cpu_nms'])

        # display a sample
        if args.plot:
            if dataset.returns_pair():
                for i, (optical, thermal,
                        prob_optical, prob_thermal,
                        mask_optical, mask_thermal) in enumerate(zip(data['optical']['image'],
                                                                     data['thermal']['image'],
                                                                     out_optical['prob'],
                                                                     out_thermal['prob'],
                                                                     data['optical']['valid_mask'],
                                                                     data['thermal']['valid_mask'],)):
                    optical = optical.squeeze().cpu()
                    thermal = thermal.squeeze().cpu()
                    prob_optical = prob_optical.squeeze().cpu()
                    prob_thermal = prob_thermal.squeeze().cpu()
                    mask_optical = mask_optical.squeeze().cpu()
                    mask_thermal = mask_thermal.squeeze().cpu()

                    if args.mask:
                        optical *= mask_optical
                        thermal *= mask_thermal

                    # convert the predictions to keypoints
                    pred_optical = torch.nonzero((prob_optical > config['prediction']['detection_threshold']).float() * mask_optical)
                    kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.numpy().astype(np.float32)]
                    pred_thermal = torch.nonzero((prob_thermal > config['prediction']['detection_threshold']).float() * mask_thermal)
                    kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.numpy().astype(np.float32)]

                    # draw predictions and ground truth on image
                    out_optical = cv2.cvtColor((np.clip(optical.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
                    out_thermal = cv2.cvtColor((np.clip(thermal.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    out_optical = cv2.drawKeypoints(out_optical,
                                                    kp_optical,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    out_thermal = cv2.drawKeypoints(out_thermal,
                                                    kp_thermal,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if 'keypoints' in data['optical'].keys() and 'keypoints' in data['thermal'].keys():
                        if data['optical']['keypoints'] is not None and data['thermal']['keypoints'] is not None:
                            kp = data['optical']['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == optical.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_optical = cv2.drawKeypoints(out_optical,
                                                            keypoints,
                                                            outImage=np.array([]),
                                                            color=(0, 0, 255),
                                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                            kp = data['thermal']['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == thermal.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_thermal = cv2.drawKeypoints(out_thermal,
                                                            keypoints,
                                                            outImage=np.array([]),
                                                            color=(0, 0, 255),
                                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # plot the raw image
                    cv2.imshow(str(i) + ' image optical', out_optical)
                    cv2.imshow(str(i) + ' prob optical', (prob_optical).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    cv2.imshow(str(i) + ' prob masked optical', (prob_optical * mask_optical).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    cv2.imshow(str(i) + ' image thermal', out_thermal)
                    cv2.imshow(str(i) + ' prob thermal', (prob_thermal).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    cv2.imshow(str(i) + ' prob masked thermal', (prob_thermal * mask_thermal).numpy() * 0.9 / config['prediction']['detection_threshold'])

            else:
                for i, (image, prob, mask) in enumerate(zip(data['image'], out['prob'], data['valid_mask'])):
                    image = image.squeeze().cpu()
                    prob = prob.squeeze().cpu()
                    mask = mask.squeeze().cpu()

                    if args.mask:
                        image *= mask

                    # convert the predictions to keypoints
                    pred = torch.nonzero((prob > config['prediction']['detection_threshold']).float() * mask)
                    predictions = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred.numpy().astype(np.float32)]

                    # draw predictions and ground truth on image
                    out_image = cv2.cvtColor((np.clip(image.numpy(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    out_image = cv2.drawKeypoints(out_image,
                                                  predictions,
                                                  outImage=np.array([]),
                                                  color=(0, 255, 0),
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    if 'keypoints' in data.keys():
                        if data['keypoints'] is not None:
                            kp = data['keypoints'][i].squeeze().cpu()

                            # convert the ground truth keypoints
                            if kp.shape == image.shape:
                                kp = torch.nonzero(kp)

                            keypoints = [cv2.KeyPoint(c[1], c[0], args.radius + 2) for c in kp.numpy().astype(np.float32)]
                            out_image = cv2.drawKeypoints(out_image,
                                                          keypoints,
                                                          outImage=np.array([]),
                                                          color=(0, 0, 255),
                                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    print(str(i) + ' is_optical: ' + str(data['is_optical'][i,0].cpu().numpy()))

                    # plot the raw image
                    cv2.imshow(str(i) + ' image', out_image)
                    cv2.imshow(str(i) + ' prob', (prob).numpy() * 0.9 / config['prediction']['detection_threshold'])
                    cv2.imshow(str(i) + ' prob masked', (prob * mask).numpy() * 0.9 / config['prediction']['detection_threshold'])
            cv2.waitKey(0)

if __name__ == "__main__":
    main()
