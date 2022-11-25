import cv2
import numpy as np
import torch
from tqdm import tqdm

from .utils import box_nms, generate_keypoint_map, data_to_device, interpolate_descriptors
from .homographies import warp_keypoints, filter_points
from .matching import get_matches

def compute_detector_metrics(net, dataloader, device, config):
    """
    Compute precision and recall and localisation error
    """
    tp, fp, prob, n_gt, dist = [], [], [], 0, []

    for data in tqdm(dataloader):
        data = data_to_device(data, device)
        out = net(data)

        pred = out['prob'] * data['valid_mask']

        if config['nms'] > 0:
            pred = box_nms(pred,
                           config['nms'],
                           config['detection_threshold'])

        out = list(map(lambda p, k: compute_tp_fp_dist(p,k), pred[:,0].cpu(), data['keypoints'].cpu()))

        for item in out:
            tp.append(item[0].tolist())
            fp.append(item[1].tolist())
            prob.append(item[2].tolist())
            n_gt += item[3]
            dist.append(item[4].tolist())

    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    prob = np.concatenate(prob)
    dist = np.concatenate(dist)

    # Sort in descending order of confidence
    sort_idx = np.argsort(prob)[::-1]
    tp = tp[sort_idx]
    fp = fp[sort_idx]
    prob = prob[sort_idx]

    # Cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, n_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, prob, dist

def compute_tp_fp_dist(prob, keypoints, zero_threshold=1e-4, distance_thresh=2.0):
    """
    Compute the true and false positive rates and the distance of the predicted to
    the ground truth keypoints for the true positives.
    """
    if prob.shape != keypoints.shape:
        keypoints = generate_keypoint_map(keypoints, prob.shape)

    kp = torch.nonzero(keypoints)

    # Filter out predictions with near-zero probability
    mask = torch.nonzero(prob > zero_threshold)
    prob = prob[mask.split(1, dim=1)]

    prob, sort_idx = torch.sort(prob, descending=True, dim=0)
    pred = mask[sort_idx.squeeze()]

    # catch the case where only one keypoint was predicted
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)

    # When several detections match the same ground truth point, only pick
    # the one with the highest score  (the others are false positive)
    diff = pred.unsqueeze(1) - kp.unsqueeze(0)
    dist =  torch.norm(diff.float(), dim=-1)
    matches = dist <= distance_thresh

    tp = []
    matched = np.zeros(len(kp), bool)
    for m in matches:
        correct = torch.any(m)
        if correct and not np.all(matched):
            gt_idx = np.argmax(m)
            tp.append(not matched[gt_idx])
            matched[gt_idx] = True
        else:
            tp.append(False)
    tp = np.array(tp, bool)
    fp = np.logical_not(tp)
    # todo compute distance from keypoints to matched gt keypoints
    return tp, fp, prob[:,0].numpy(), len(kp), dist[matches].numpy()

def compute_mAP(precision, recall):
    """
    Compute average precision.
    """
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))

def compute_repeatability_multispectral(net, dataloader, device, config,
                                        distance_thresh=3, verbose=False):
    """
    Compute the repeatability of the keypoints
    """

    repeatability = []
    n_kp_optical = []
    n_kp_thermal = []

    for data in tqdm(dataloader):
        detection_threshold = config['prediction']['detection_threshold']
        # extract homographies
        if 'homography' in data['optical'].keys():
            H_optical = data['optical']['homography']
        else:
            H_optical = torch.eye(3,3).repeat(data['optical']['image'].shape[0],1,1)

        if 'homography' in data['thermal'].keys():
            H_thermal = data['thermal']['homography']
        else:
            H_thermal = torch.eye(3,3).repeat(data['thermal']['image'].shape[0],1,1)

        # get the predictions
        data = data_to_device(data, device)
        out_optical = net(data['optical'])
        out_thermal = net(data['thermal'])

        # calculate nms prob if requested
        if config['prediction']['nms'] > 0:
            out_optical['prob'] = box_nms(out_optical['prob'],
                                          config['prediction']['nms'],
                                          detection_threshold,
                                          keep_top_k=config['prediction']['topk'],
                                          on_cpu=config['prediction']['cpu_nms'])

            out_thermal['prob'] = box_nms(out_thermal['prob'],
                                          config['prediction']['nms'],
                                          detection_threshold,
                                          keep_top_k=config['prediction']['topk'],
                                          on_cpu=config['prediction']['cpu_nms'])

        # process each sample individually
        for (prob_o, prob_t, mask_o,
             mask_t, h_o, h_t) in zip(out_optical['prob'].split(1),
                                      out_thermal['prob'].split(1),
                                      data['optical']['valid_mask'].split(1),
                                      data['thermal']['valid_mask'].split(1),
                                      H_optical.split(1),
                                      H_thermal.split(1)):

            # get keypoints
            kp_optical = torch.nonzero((prob_o.squeeze() > detection_threshold).float() * mask_o.squeeze())
            kp_thermal = torch.nonzero((prob_t.squeeze() > detection_threshold).float() * mask_t.squeeze())

            n_kp_optical.append(kp_optical.shape[0])
            n_kp_thermal.append(kp_thermal.shape[0])

            # convert to numpy
            kp_thermal = kp_thermal.cpu().numpy()
            kp_optical = kp_optical.cpu().numpy()
            image_shape = prob_o.squeeze().cpu().numpy().shape

            # warp optical images to the thermal frame
            warped_optical = warp_keypoints(kp_optical, h_o.squeeze().inverse().cpu().numpy())
            warped_optical = warp_keypoints(warped_optical, h_t.squeeze().cpu().numpy())
            warped_optical = filter_points(warped_optical, image_shape)

            # warp thermal images to the optical frame
            warped_thermal = warp_keypoints(kp_thermal, h_t.squeeze().inverse().cpu().numpy())
            warped_thermal = warp_keypoints(warped_thermal, h_o.squeeze().cpu().numpy())
            warped_thermal = filter_points(warped_thermal, image_shape)

            # compute the repeatability
            N_thermal = warped_thermal.shape[0]
            N_optical = warped_optical.shape[0]

            warped_thermal = np.expand_dims(warped_thermal, 1)
            warped_optical = np.expand_dims(warped_optical, 1)
            kp_optical = np.expand_dims(kp_optical, 0)
            kp_thermal = np.expand_dims(kp_thermal, 0)

            dist1 = np.linalg.norm(warped_thermal - kp_optical, ord=None, axis=2)
            dist2 = np.linalg.norm(warped_optical - kp_thermal, ord=None, axis=2)

            count1 = 0
            count2 = 0
            if kp_optical.shape[1] != 0:
                min1 = np.min(dist1, axis=1)
                count1 = np.sum(min1 <= distance_thresh)
            if kp_thermal.shape[1] != 0:
                min2 = np.min(dist2, axis=1)
                count2 = np.sum(min2 <= distance_thresh)
            if N_thermal + N_optical > 0:
                repeatability.append((count1 + count2) / (N_thermal + N_optical))

    return np.mean(repeatability), repeatability, n_kp_optical, n_kp_thermal

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

def compute_descriptor_metrics(net, dataloader, device, config, threshold_keypoints, threshold_warp):
    """
    Compute precision and recall and localisation error
    """
    tp_optical = []
    distance_optical = []
    tp_thermal = []
    distance_thermal = []
    n_gt_optical = 0
    n_gt_thermal = 0
    m_score_optical = []
    m_score_thermal = []
    pts_dist = []

    for data in tqdm(dataloader):
        # predict
        data = data_to_device(data, device)
        out_optical = net(data['optical'])
        out_thermal = net(data['thermal'])

        # mask the output
        prob_optical = out_optical['prob'] * data['optical']['valid_mask']
        prob_thermal = out_thermal['prob'] * data['thermal']['valid_mask']

        if config['nms'] > 0:
            prob_thermal = box_nms(prob_thermal,
                                   config['nms'],
                                   config['detection_threshold'],
                                   keep_top_k = config['topk'],
                                   on_cpu=config['cpu_nms'])
            prob_optical = box_nms(prob_optical,
                                   config['nms'],
                                   config['detection_threshold'],
                                   keep_top_k = config['topk'],
                                   on_cpu=config['cpu_nms'])

        # add identity homography to data if not present
        if 'homography' not in data['optical'].keys():
            data['optical']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).repeat(data['optical']['image'].shape[0],1,1)

        if 'homography' not in data['thermal'].keys():
            data['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).repeat(data['optical']['image'].shape[0],1,1)

        for (prob_o, prob_t,
             h_o, h_t,
             desc_o, desc_t) in zip(prob_optical, prob_thermal,
                                    data['optical']['homography'], data['thermal']['homography'],
                                    out_optical['desc'], out_thermal['desc']):
            # get the combined homography from optical to thermal
            gt_homography = torch.mm(h_t,h_o.inverse())

            # compute keypoints
            pred_optical = torch.nonzero((prob_o.squeeze() > config['detection_threshold']).float())
            pred_thermal = torch.nonzero((prob_t.squeeze() > config['detection_threshold']).float())

            # get the descriptors
            H_o, W_o = data['optical']['image'].shape[2:]
            H_t, W_t = data['thermal']['image'].shape[2:]
            desc_optical= interpolate_descriptors(pred_optical, desc_o, H_o, W_o)
            desc_thermal = interpolate_descriptors(pred_thermal, desc_t, H_t, W_t)

            # match the keypoints
            if desc_optical.shape[0] > 0 and desc_thermal.shape[0] > 0:
                matches_thermal = get_matches(desc_thermal.cpu().numpy(),
                                              desc_optical.cpu().numpy(),
                                              'bfmatcher',
                                              False,
                                              crossCheck = True)
                matches_optical = get_matches(desc_optical.cpu().numpy(),
                                              desc_thermal.cpu().numpy(),
                                              'bfmatcher',
                                              False,
                                              crossCheck = True)
            else:
                matches_thermal = []
                matches_optical = []

            matches_optical = sorted(matches_optical, key = lambda x:x.distance)
            matches_thermal = sorted(matches_thermal, key = lambda x:x.distance)

            # warp the keypoints to get the ground truth position
            warped_optical = warp_keypoints(pred_optical.cpu().float().numpy(), gt_homography.cpu().numpy(), np.float)
            warped_thermal = warp_keypoints(pred_thermal.cpu().float().numpy(), gt_homography.inverse().cpu().numpy(), np.float)

            # compute the correct matches matrix
            dist = torch.from_numpy(warped_optical).to(pred_thermal.device).unsqueeze(1) - pred_thermal.unsqueeze(0)
            correct_optical = torch.norm(dist.float(), dim=-1) <= threshold_keypoints
            dist = torch.from_numpy(warped_thermal).to(pred_thermal.device).unsqueeze(1) - pred_optical.unsqueeze(0)
            correct_thermal = torch.norm(dist.float(), dim=-1) <= threshold_keypoints

            # number of possible matches (at least one valid kp in the thermal spectrum for each one from the optical spectrum)
            n_gt_optical += correct_optical.sum(1).nonzero().shape[0]
            n_gt_thermal += correct_thermal.sum(1).nonzero().shape[0]

            # check if the matches from the matcher are true or false positives
            num_matched_optical = 0
            for m in matches_optical:
                num_matched_optical += correct_optical[m.queryIdx, m.trainIdx].item()
                tp_optical.append(correct_optical[m.queryIdx, m.trainIdx].item())
                distance_optical.append(m.distance)

            num_matched_thermal = 0
            for m in matches_thermal:
                num_matched_thermal += correct_thermal[m.queryIdx, m.trainIdx].item()
                tp_thermal.append(correct_thermal[m.queryIdx, m.trainIdx].item())
                distance_thermal.append(m.distance)

            # compute the m-score (number of recovered keypoints over possible keypoints)
            image_shape = prob_o.squeeze().cpu().numpy().shape
            N_optical = filter_points(warped_optical, image_shape).shape[0]
            N_thermal = filter_points(warped_thermal, image_shape).shape[0]
            if N_optical > 0:
                m_score_optical.append(float(num_matched_optical)/N_optical)
            else:
                m_score_optical.append(0.0)
            if N_thermal > 0:
                m_score_thermal.append(float(num_matched_thermal)/N_thermal)
            else:
                m_score_thermal.append(0.0)

            # estimate the homography
            if desc_optical.shape[0] > 0 and desc_thermal.shape[0] > 0:
                matches = get_matches(desc_optical.cpu().numpy(),
                                      desc_thermal.cpu().numpy(),
                                      config['matching']['method'],
                                      config['matching']['knn_matches'],
                                      **config['matching']['method_kwargs'])
            else:
                matches = []

            kp_optical = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_optical.cpu().numpy().astype(np.float32)]
            kp_thermal = [cv2.KeyPoint(c[1], c[0], 1) for c in pred_thermal.cpu().numpy().astype(np.float32)]
            optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            if optical_pts.shape[0] < 4 or thermal_pts.shape[0] < 4:
                H_est = None
                matchesMask = []
            else:
                H_est, mask = cv2.findHomography(optical_pts, thermal_pts, cv2.RANSAC, ransacReprojThreshold=config['reprojection_threshold'])

            # compute the homography correctness
            if H_est is not None:
                pts = np.array([[0, 0], [H_o, 0], [0, W_o], [H_o, H_o]])
                pts_warped_gt = warp_keypoints(pts, gt_homography.cpu().numpy(), np.float)
                pts_warped_est = warp_keypoints(pts, H_est, np.float)
                pts_dist.append(np.linalg.norm(pts_warped_est - pts_warped_gt, axis=1).sum()/4)
            else:
                pts_dist.append(999.0)

    # convert to numpy arrays
    tp_optical = np.array(tp_optical)
    distance_optical = np.array(distance_optical)
    m_score_optical = np.array(m_score_optical)

    tp_thermal = np.array(tp_thermal)
    distance_thermal = np.array(distance_thermal)
    m_score_thermal = np.array(m_score_thermal)

    pts_dist = np.array(pts_dist)

    # sort in ascending order of distance
    sort_idx_optical = np.argsort(distance_optical)
    tp_optical = tp_optical[sort_idx_optical]
    fp_optical = np.logical_not(tp_optical)
    distance_optical = distance_optical[sort_idx_optical]

    sort_idx_thermal = np.argsort(distance_thermal)
    tp_thermal = tp_thermal[sort_idx_thermal]
    fp_thermal = np.logical_not(tp_thermal)
    distance_thermal = distance_thermal[sort_idx_thermal]

    # compute the precision and recall
    tp_optical_cum = np.cumsum(tp_optical)
    tp_thermal_cum = np.cumsum(tp_thermal)
    fp_optical_cum = np.cumsum(fp_optical)
    fp_thermal_cum = np.cumsum(fp_thermal)

    recall_optical = div0(tp_optical_cum, n_gt_optical)
    recall_thermal = div0(tp_thermal_cum, n_gt_thermal)

    precision_optical = div0(tp_optical_cum, tp_optical_cum + fp_optical_cum)
    precision_thermal = div0(tp_thermal_cum, tp_thermal_cum + fp_thermal_cum)

    recall_optical = np.concatenate([[0], recall_optical, [1]])
    precision_optical = np.concatenate([[0], precision_optical, [0]])
    precision_optical = np.maximum.accumulate(precision_optical[::-1])[::-1]

    recall_thermal = np.concatenate([[0], recall_thermal, [1]])
    precision_thermal = np.concatenate([[0], precision_thermal, [0]])
    precision_thermal = np.maximum.accumulate(precision_thermal[::-1])[::-1]

    # compute nearest neighbor mean average precision
    nn_map_optical = compute_mAP(precision_optical, recall_optical)
    nn_map_thermal = compute_mAP(precision_thermal, recall_thermal)
    nn_map = (nn_map_optical + nn_map_thermal) * 0.5

    # compute the matching score
    m_score = (m_score_optical.mean() + m_score_thermal.mean()) * 0.5

    # compute homography estimation accuracy
    average_h_error = pts_dist.mean()
    h_correctness = (pts_dist < threshold_warp).sum() / len(pts_dist)

    # create out dictionary
    out = {
        'tp_optical': tp_optical,
        'tp_thermal': tp_thermal,
        'fp_optical': fp_optical,
        'fp_thermal': fp_thermal,
        'distance_optical': distance_optical,
        'distance_thermal': distance_thermal,
        'recall_optical': recall_optical,
        'recall_thermal': recall_thermal,
        'precision_optical': precision_optical,
        'precision_thermal': precision_thermal,
        'nn_map_optical': nn_map_optical,
        'nn_map_thermal': nn_map_thermal,
        'nn_map': nn_map,
        'm_score_optical': m_score_optical,
        'm_score_thermal': m_score_thermal,
        'm_score': m_score,
        'pts_dist': pts_dist,
        'average_h_error': average_h_error,
        'h_correctness': h_correctness,
        }

    return out
