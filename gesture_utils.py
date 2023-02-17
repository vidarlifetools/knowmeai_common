import numpy as np
import pickle
import vg
import math
from constants import *
import mediapipe as mp
from feature_constants import \
    mp_pose_min_detection_confidence,\
    mp_pose_min_tracking_confidence
from feature_constants import keypoint_mapping_table


# Based on recognize_gesture.py in machine-learning
class PoseFeature:
    def __init__(self):
        # TODO Initiate tha class depending on static_image _mode
        self.mp_pose = mp.solutions.pose.Pose(
            # static_image_mode=static_image_mode,
            # model_complexity=self.config.model_complexity,
            # enable_segmentation=self.config.enable_segmentation,
            min_detection_confidence=mp_pose_min_detection_confidence,
            min_tracking_confidence=mp_pose_min_tracking_confidence
        )

    def get(self, image):
        results = self.mp_pose.process(image)
        if results.pose_world_landmarks.landmark:
            pose_3d = np.zeros((17, 5), dtype=np.float32)
            for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                if keypoint_mapping_table[i] != -1:
                    conf = landmark.visibility if (keypoint_mapping_table[i] > 12 and landmark.visibility > 0.8) or (
                                keypoint_mapping_table[i] <= 12) else 0.0
                    pose_3d[keypoint_mapping_table[i], :] = [landmark.x * 1000.0, landmark.y * 1000.0, landmark.z * 1000.0,
                                                    conf, 0.0]
            return pose_3d, True, image
        else:
            return None, False, image

#from utilities.recognize_gesture import compute_feature_vector_pose,\
#    predict_pose_class, compute_feature_vector_gesture, initialize_buffer_info,\
#    get_sequence_from_buffer_info, update_buffer_info
class GesturePrediction:
    def __init__(self, model_filename_pose, model_filename_gesture, logger):
        with open(model_filename_pose, "rb") as file:
            self.pose_model = pickle.load(file)
        with open(model_filename_gesture, "rb") as file:
            self.gesture_model = pickle.load(file)
        self.nof_pose_classes = len(self.pose_model.classes_)
        self.logger = logger
        self.buffer_info = initialize_buffer_info()

    def get_class(self, keypoints):

        # If key point flag is invalid or if fts are invalid, pose label is missing
        pose_class = MISSING_CLASS_POSE

        # Compute pose features
        fts, found_fts = compute_feature_vector_pose(keypoints)

        # If features are valid, predict pose label and log information
        if found_fts:
            pose_class = predict_pose_class(self.pose_model, fts)
        # Update pose labels buffer
        update_buffer_info(pose_class, self.buffer_info)

        # Log nr of missing pose labels
        nof_missing = len(self.buffer_info['pose_class_buffer'][self.buffer_info['pose_class_buffer'] == MISSING_CLASS_POSE])
        if nof_missing > 0:
            self.logger.debug('WARNING_GESTURE_SPURIOUS_SKELETON: timestamp '  +
                         ' missing frames in current buffer: ' + str(nof_missing))

        # If length of buffer exceeds GESTURE_BUFFER_LEN
        if self.buffer_info['filled_buffer']:
            pose_class_sequence = get_sequence_from_buffer_info(self.buffer_info)

            # Compute gesture feature vector, and gesture class if feature vector is valid
            fts, fts_valid = compute_feature_vector_gesture(pose_class_sequence, self.nof_pose_classes)
            if fts_valid:
                class_nos = np.array(self.gesture_model.predict(fts.reshape(1, -1)), dtype=np.uint8)

                # Log predicted gesture class
                self.logger.debug(f'recognize_gesture_thread: timestamp  gesture_fts {list(fts)} '
                             f'gesture_class {class_nos[0]}')

            # If fts are invalid, gesture labels are missing
            else:
                class_nos = np.array([MISSING_CLASS_GESTURE], dtype=np.uint8)

        # If buffer is not filled, gesture label is missing
        else:
            class_nos = np.array([MISSING_CLASS_GESTURE], dtype=np.uint8)

        # Log results
        self.logger.debug(f'recognize_gesture_thread:  predicted class {class_nos[0]}')

        class_probs = np.array([-1.0], dtype=np.float32)
        # If thread is not ended
        nof_results = np.array([len(class_nos)], dtype=np.uint8)

        return class_nos, class_probs



# TODO buffer as long vector rather than constantly changing short vector?
def initialize_buffer_info():
    return {'pose_class_buffer': np.zeros(GESTURE_BUFFER_LEN, dtype=int),
            'filled_buffer': False,
            'ind': 0}


# TODO check update script -> new values should be added at the end of buffer, otherwise consecutive missing class count is off?
def update_buffer_info(pose_class, buffer_info):

    buffer_info['pose_class_buffer'][buffer_info['ind']] = pose_class

    buffer_info['ind'] += 1
    if buffer_info['ind'] == GESTURE_BUFFER_LEN:
        buffer_info['ind'] = 0
        buffer_info['filled_buffer'] = True


def get_sequence_from_buffer_info(buffer_info):

    ind = buffer_info['ind']
    buffer = buffer_info['pose_class_buffer']

    if ind==0:
        return buffer
    else:
        return np.concatenate((buffer[ind:], buffer[:ind]))


# TODO more readable?
def compute_max_consecutive_missing_class_pose(sequence):
    # Determine the groups of missing class poses in the sequence and find whether there are consecutive elements
    # within the groups
    data_missing_class_pose = np.where(sequence == MISSING_CLASS_POSE)[0]
    consecutive_missing_class_pose = np.split(data_missing_class_pose, np.where(np.diff(data_missing_class_pose) != 1)[0]+1)
    max_consecutive_missing_class_pose = max([len(consecutive_missing_class_pose[i]) for i in range(len(consecutive_missing_class_pose))])

    return consecutive_missing_class_pose, max_consecutive_missing_class_pose


def compute_feature_vector_pose(keypoints_in, normalize_features=True):
    # normalize_features = (keypoints_in.shape[0] == NOF_KEYPOINTS) # normalize if selected NR method
    keypoints = preprocess_keypoints(keypoints_in)

    if not keypoints_are_OK(keypoints):
        fts = []
        found_fts = False
    else:
        found_fts = True
        if normalize_features:
            dist_const = compute_distance(keypoints, [(Neck, MidHip)])[0] # Normalize using 1.5*distance from mid hip to neck

            if dist_const == MISSING_POSE_FT_VALUE:
                fts = []
                found_fts = False
            else:
                fts = compute_distance(keypoints, TUPPLE_DIST, dist_const=dist_const * 1.5) + \
                      compute_angle(keypoints, TUPPLE_ANGLE, normalize=normalize_features)
        else:
            fts = compute_distance(keypoints, TUPPLE_DIST, dist_const=None) + \
                  compute_angle(keypoints, TUPPLE_ANGLE, normalize=normalize_features)

    return fts, found_fts


def preprocess_keypoints(keypoints_in, input_dimensions=3):
    if keypoints_in.shape[0] != NOF_KEYPOINTS:
        # Selected EmTech method for extracting keypoints
        keypoints = np.zeros((NOF_KEYPOINTS, input_dimensions+1), dtype=keypoints_in.dtype)

        keypoints[Nose, :] = keypoints_in[nose, :4]
        keypoints[LEye, :] = keypoints_in[left_eye, :4]
        keypoints[REye, :] = keypoints_in[right_eye, :4]
        keypoints[LEar, :] = keypoints_in[left_ear, :4]
        keypoints[REar, :] = keypoints_in[right_ear, :4]
        keypoints[LShoulder, :] = keypoints_in[left_shoulder, :4]
        keypoints[RShoulder, :] = keypoints_in[right_shoulder, :4]
        keypoints[LElbow, :] = keypoints_in[left_elbow, :4]
        keypoints[RElbow, :] = keypoints_in[right_elbow, :4]
        keypoints[LWrist, :] = keypoints_in[left_wrist, :4]
        keypoints[RWrist, :] = keypoints_in[right_wrist, :4]
        keypoints[LHip, :] = keypoints_in[left_hip, :4]
        keypoints[RHip, :] = keypoints_in[right_hip, :4]
        keypoints[LKnee, :] = keypoints_in[left_knee, :4]
        keypoints[RKnee, :] = keypoints_in[right_knee, :4]
        keypoints[LAnkle, :] = keypoints_in[left_ankle, :4]
        keypoints[RAnkle, :] = keypoints_in[right_ankle, :4]

        keypoints[Neck, ] = get_mid(keypoints[RShoulder, ], keypoints[LShoulder, ])
        keypoints[MidHip, ] = get_mid(keypoints[RHip, ], keypoints[LHip, ])
    else:
        keypoints = keypoints_in

    return keypoints

def get_mid(kp1, kp2):
    if kp1[-1] > CONFIDENCE_LIMIT_KEYPOINT and kp2[-1] > CONFIDENCE_LIMIT_KEYPOINT:
        return 0.5 * (kp1 + kp2)
    else:
        return 0.0


def keypoints_are_OK(keypoints):
    used_keypoints = np.unique(np.array(
        [item for sublist in TUPPLE_ANGLE for item in sublist] + [item for sublist in TUPPLE_DIST for item in sublist]))
    keypoint_OK = np.zeros(used_keypoints.shape[0])
    for i, k in enumerate(used_keypoints):
        if keypoints[k, 3] > CONFIDENCE_LIMIT_KEYPOINT:
            keypoint_OK[i] = 1

    frac_OK_keypoints = np.sum(keypoint_OK) / float(keypoint_OK.shape[0])

    return frac_OK_keypoints > MIN_FRAC_OK_KEYPOINTS


def compute_distance(keypoints, tupple_dist, dist_const=None):
    dist = []
    for a, b in tupple_dist:
        if keypoints[a, 3] > CONFIDENCE_LIMIT_KEYPOINT and keypoints[b, 3] > CONFIDENCE_LIMIT_KEYPOINT:
            d = np.linalg.norm(keypoints[a, 0:3] - keypoints[b, 0:3])
            if dist_const is not None:
                d = d/dist_const

        else:  # distance set to MISSING because low confidence for at least one keypoint
            d = MISSING_POSE_FT_VALUE

        dist += [d]

    return dist


def compute_angle(keypoints, tupple_angle, normalize=False):
    angle = []
    for a, b, c in tupple_angle:
        if (keypoints[a, 3] < CONFIDENCE_LIMIT_KEYPOINT or keypoints[b, 3] < CONFIDENCE_LIMIT_KEYPOINT) or \
                keypoints[c, 3] < CONFIDENCE_LIMIT_KEYPOINT:
            ang = MISSING_POSE_FT_VALUE
        else:
            vector1 = keypoints[a, 0:3] - keypoints[b, 0:3]
            vector2 = keypoints[c, 0:3] - keypoints[b, 0:3]

            if sum(np.abs(np.array(vector1))) > 0 and sum(np.abs(np.array(vector2))) > 0:
                ang = vg.signed_angle(vector1, vector2, look=vg.basis.z, units='rad')
                ang = ang + 2 * math.pi if ang < 0 else ang

                if normalize:
                    ang /= 2 * math.pi
            else:
                ang = MISSING_POSE_FT_VALUE

        angle += [ang]

    return angle


def predict_pose_class(model, fts):
    class_no = model.predict(np.array(fts).reshape(1, -1))
    return class_no


def compute_feature_vector_gesture(pose_class_sequence, nof_pose_classes):
    """ Calculate gesture feature vector from pose labels sequence """
    # If too many consecutive pose labels are missing, the gesture feature vector is not valid
    _, max_consecutive_missing_class_pose = compute_max_consecutive_missing_class_pose(pose_class_sequence)
    if max_consecutive_missing_class_pose > MAX_CONSECUTIVE_MISSING_CLASS_POSE:
        return [], False
    else:
        valid_pose_class_sequence = pose_class_sequence[pose_class_sequence != MISSING_CLASS_POSE]
        fts = np.histogram(valid_pose_class_sequence, bins=np.arange(0, nof_pose_classes+1))[0]
        return fts, True
