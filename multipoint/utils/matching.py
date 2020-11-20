import cv2
import numpy as np

def get_matches(desc_1, desc_2, method = 'bfmatcher', knn_matches = False, **kwargs):
    # match the keypoints
    if method == 'bfmatcher':
        matcher = cv2.BFMatcher(cv2.NORM_L2, **kwargs)

    elif method == 'flann':
        matcher = cv2.FlannBasedMatcher()

    elif method == 'nnmatcher':
        matcher = NNMatcher(**kwargs)

    elif method == 'thresholdmatcher':
        matcher = ThresholdMatcher(**kwargs)

    else:
        raise ValueError('unknown matching method')

    if knn_matches:
        all_matches = matcher.knnMatch(desc_1, desc_2, 2)
        #-- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.9
        matches = []
        for m,n in all_matches:
            if m.distance < ratio_thresh * n.distance:
                matches.append(m)

    else:
        matches = matcher.match(desc_1, desc_2)

    return matches

class NNMatcher():
    def __init__(self, threshold=0.7):
        self.nn_thresh = threshold
        if threshold < 0.0:
            raise ValueError('\'threshold\' should be non-negative')

    def match(self, desc1, desc2):
        desc1 = desc1.transpose()
        desc2 = desc2.transpose()

        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return []
 
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < self.nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx

        # Populate the final 3xN match data structure.
        matches = []
        for i1, i2, d in zip(m_idx1, m_idx2, scores):
            matches.append(cv2.DMatch(i1, i2, d))

        return matches

class ThresholdMatcher():
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        if threshold < 0.0:
            raise ValueError('\'threshold\' should be non-negative')

    def match(self, desc1, desc2):
        desc1 = desc1.transpose()
        desc2 = desc2.transpose()

        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return []
 
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))

        # get matches and scores
        idx = np.argwhere(dmat < self.threshold)

        matches = []
        for i in idx:
            matches.append(cv2.DMatch(i[0], i[1], dmat[tuple(i)]))

        return matches
