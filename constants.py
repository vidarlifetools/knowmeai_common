# === For testing that feature vectors computed during labelling, training and testing are equal:
#        set WRITE_FEATURES_LABEL_TRAIN_TEST to True and
#        set WRITE_FEATURES_LABEL_TRAIN_TEST_PATH to an existing, empty catalogue
WRITE_FEATURES_LABEL_TRAIN_TEST  = False
WRITE_FEATURES_LABEL_TRAIN_TEST_PATH = "" #WRITE_FEATURES_LABEL_TRAIN_TEST_PATH = '/nr/common/sambadata1/pro/LIFETOOLS/Usr/Marit/Annotations_March2022_mediapipe/lukas/machine-learning/Fts_files'


# === For test purposes: Select whether to include sound, gesture and face in the feature vector used for compound recognition
INCLUDE_FACE_EXPR = True
INCLUDE_GESTURE   = True
INCLUDE_SOUND     = False


# === Face crops and skeletons
from dataclasses import dataclass

@dataclass
class FaceData:
    valid: bool
    landmarks: list
    bbox_person: list


@dataclass
class SkeletonData:
    valid: bool
    keypoints: list
    bbox_person: list


@dataclass
class SoundData:
    valid: bool
    feature: list

# === Sound classification constants
CLIENT_SOUND_CLASSES    = 10    # Number of classes for client sound features
SOUND_CLASSES           = 10    # Number of final sound classes (after histogram manipulation)
NON_CLIENT_CLASS        = 10    # Must be higher than max client sound class
CLIENT_MLP_LAYER1_SIZE  = 50
CLIENT_MLP_LAYER2_SIZE  = 20
SPLIT_MLP_LAYER1_SIZE   = 50
SPLIT_MLP_LAYER2_SIZE   = 10
SOUND_MLP_LAYER1_SIZE   = 50
SOUND_MLP_LAYER2_SIZE   = 10
SOUND_HISTOGRAM_DEPTH   = 6     # Number of histograms used for sound tensor generation
CLIENT_SOUND_THRESHOLD  = 3.0   # The sum of all elements in the sound vector must be grater than this to be considered client sound

# === Classification constants
MISSING_CLASS          = 255
MISSING_CLASS_GESTURE  = 254
MISSING_CLASS_FACE     = 253
MISSING_CLASS_POSE     = 252
MISSING_CLASS_SOUND    = 251

MAX_CONSECUTIVE_MISSING_CLASS_POSE = 2  # TODO: Decide value for MAX_CONSECUTIVE_MISSING_CLASS_POSE. Should probably depend on frame rate.
GESTURE_BUFFER_LEN                 = 15 # TODO: Decide value for GESTURE_BUFFER_LEN. Should probably depend on frame rate.
SOUND_BUFFER_LEN                   = 1  # TODO: Decide value for SOUND_BUFFER_LEN. Should probably depend on frame rate.
COMPOUND_BUFFER_LEN                = 30 # TODO: Decide value for COMPOUND_BUFFER_LEN. Should probably depend on frame rate.
NOF_INTERVALS_COMPOUND             = 5  # TODO: Decide value for NOF_INTERVALS_COMPOUND.
VALID_MISSING_PORTION              = 0.5 # Portion of missing sub-labels (face, gesture, sound) for a compound feature vector to still be valid
CHUNK_PREDICTION                   = 1  # Prediction (compound recognition)


# === Labelling and training constants
LABEL_AND_TRAIN_ALL = 0 # initialize, label expression, label and train poses, label gestures, train expression, train gesture, train compound
LABEL_EXPR          = 1 # label expression, train expression, train compound
LABEL_POSE          = 2 # label and train poses, label gestures, train gesture, train compound
LABEL_GEST          = 3 # label gestures, train gesture, train compound
TRAIN_ALL           = 4 # train expression, train gesture, train compound
TRAIN_EXPR          = 5 # train expression, train compound
TRAIN_GEST          = 6 # train gesture, train compound
TRAIN_COMPOUND      = 7 # train compound


# === Name of catalogues in the catalogue for a person
RAW_CAT         = 'raw' # Catalogue for video and depth files
ANNOTATION_CAT  = 'annotation'  # Catalogue with manually annotated json files
FACE_CROPS_CAT  = 'face'
SKELETONS_CAT   = 'skeleton'
SOUND_CAT       = 'sound'
ML_CAT          = 'machine-learning' # Catalogue with output from the machine-learning module


# === Not person specific json files with label and context information
LABELS_FILENAME     = 'labels.json'
CONTEXT_FILENAME    = 'context.json'


# === Json files with information about models and training data for one person
XREF_FILE                   = 'xref.json'
ANNOTATED_LABELS_FILENAME   = 'labels_annotated.json'
ANNOTATED_PERSON_FILENAME   = 'person_annotated.json'
LABELLED_LABELS_FILENAME    = 'labels_labelled.json'
LABELLED_PERSON_FILENAME    = 'person_labelled.json'


# === Filenames for face crops and skeletons
TRAIN_DICT_FILENAME_EXPRESSION   = 'train_expr.json'
TRAIN_DICT_FILENAME_POSE         = 'train_pose.json'
TRAIN_DICT_FILENAME_GESTURE      = 'train_gest.json'
TRAIN_DICT_FILENAME_SOUND        = 'train_sound.json'
TRAIN_DICT_FILENAME_COMPOUND     = 'train_comp.json'


# === Models for person
CATALOGUE_NAME_MODELS       = 'Models'
MODEL_FILENAME_EXPRESSION   = 'svm_expression.pkl'
MODEL_FILENAME_POSE         = 'svm_pose.pkl'
MODEL_FILENAME_GESTURE      = 'svm_gesture.pkl'
MODEL_FILENAME_CLIENT        = 'mlp_client.pkl'     # The annotated client feature class
MODEL_FILENAME_SOUND        = 'mlp_sound.pkl'       # The final sound class
MODEL_FILENAME_SPLIT_SOUND  = 'mlp_split_sound.pkl' # The model to separate client sound from all other sound
MODEL_FILENAME_COMPOUND     = 'svm_compound.pkl'


# === Gesture features for person
GESTURE_FEATURES_FILENAME   = 'gesture_fts.npy'


# === CSV files with data characteristics
MANUAL_ANNOTATIONS_CSV  = 'Manual_annotations.csv'
ANNOTATIONS_CSV         = 'Annotations.csv'


# === Files with overview over video characteristics
OVERVIEW_VIDEOS = 'Overview_videos'


# === ZMQ communication
FACE_TOPIC      = 'FACE'
SKELETON_TOPIC  = 'SKELETON'
SOUND_TOPIC     = 'SOUND'
FACE_EXPR_TOPIC = 'FACE_EXPR'
GESTURE_TOPIC   = 'GESTURE'
COMPOUND_TOPIC  = 'COMPOUND'

B_FACE_TOPIC      = b'FACE'
B_SKELETON_TOPIC  = b'SKELETON'
B_SOUND_TOPIC     = b'SOUND'
B_FACE_EXPR_TOPIC = b'FACE_EXPR'
B_GESTURE_TOPIC   = b'GESTURE'
B_COMPOUND_TOPIC  = b'COMPOUND'

NOT_OK = 0
OK     = 1


# === Skeleton
CONFIDENCE_LIMIT_KEYPOINT = 0.5
MIN_FRAC_OK_KEYPOINTS     = 0.5
MISSING_POSE_FT_VALUE     = 0.0 # If a pose feature cannot be calculated due to missing key points, it's replaced with this value
SKELETON_DIMENSIONS = (17, 5)

# There are in total 25 body-parts ordered according to the output of OpenPose
Nose          = 0
Neck          = 1
RShoulder     = 2
RElbow        = 3
RWrist        = 4
LShoulder     = 5
LElbow        = 6
LWrist        = 7
MidHip        = 8
RHip          = 9
RKnee         = 10
RAnkle        = 11
LHip          = 12
LKnee         = 13
LAnkle        = 14
REye          = 15
LEye          = 16
REar          = 17
LEar          = 18
LBigToe       = 19
LSmallToe     = 20
LHeel         = 21
RBigToe       = 22
RSmallToe     = 23
RHeel         = 24
NOF_KEYPOINTS = 25

# Selected angles
TUPPLE_ANGLE = [(Nose, Neck, RShoulder),
                (Neck, RShoulder, RElbow),
                (RShoulder, RElbow, RWrist),
                (Nose, Neck, LShoulder),
                (Neck, LShoulder, LElbow),
                (LShoulder, LElbow, LWrist),
                (MidHip, Neck, RElbow),
                (MidHip, Neck, LElbow),
                (MidHip, Neck, RWrist),
                (MidHip, Neck, LWrist)
                ]

# Selected distances
TUPPLE_DIST = [(Nose, MidHip),
               (RElbow, MidHip),
               (RWrist, MidHip),
               (LElbow, MidHip),
               (LWrist, MidHip),
               (Nose, LWrist),
               (Nose, RWrist),
               (Nose, LElbow),
               (Nose, RElbow)
               ]

# COCO_KEYPOINT_INDEXES - There are in total 17 body-parts ordered according to the skeletons of the current method for extracting skeletons
nose           = 0
left_eye       = 1
right_eye      = 2
left_ear       = 3
right_ear      = 4
left_shoulder  = 5
right_shoulder = 6
left_elbow     = 7
right_elbow    = 8
left_wrist     = 9
right_wrist    = 10
left_hip       = 11
right_hip      = 12
left_knee      = 13
right_knee     = 14
left_ankle     = 15
right_ankle    = 16


# === Face landmarks
NOF_FACE_LANDMARKS = 468
MISSING_LANDMARK_FT_VALUE = 0.0 # If a landmark feature cannot be calculated, it's replaced with this value

mediapipe_landmarks = [
    61,     # 00 Mouth end (right)
    292,    # 01 Mouth end (left)
	0,      # 02 Upper lip (middle)
	17,	    # 03 Lower lip (middle)
	50,	    # 04 Right cheek
	280,	# 05 Left cheek
	48,	    # 06 Nose right end
	4,	    # 07 Nose tip
	289,	# 08 Nose left end
	206,	# 09 Upper jaw (right)
	426,	# 10 Upper jaw (left)
	133,	# 11 Right eye (inner)
	130,	# 12 Right eye (outer)
	159,	# 13 Right upper eyelid (middle)
	145,	# 14 Right lower eyelid (middle)
	362,	# 15 Left eye (inner)
	359,	# 16 Left eye (outer)
	386,	# 17 Left upper eyelid (middle)
	374,	# 18 Left lower eyelid (middle)
	122,	# 19 Nose bridge (right)
	351,	# 20 Nose bridge (left)
	46,	    # 21 Right eyebrow (outer)
	105,	# 22 Right eyebrow (middle)
	107,	# 23 Right eyebrow (inner)
	276,	# 24 Left eyebrow (outer)
	334,	# 25 Left eyebrow (middle)
	336,	# 26 Left eyebrow (inner)
]

mediapipe_angles = [(2, 0, 3),
                    (0, 2, 1),
                    (6, 7, 8),
                    (9, 7, 10),
                    (0, 7, 1),
                    (1, 5, 8),
                    (1, 10, 8),
                    (13, 12, 14),
                    (21, 22, 23),
                    (6, 19, 23),
                    ]
