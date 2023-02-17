# Mediapipe face parameters
mp_face_static_mode = False,
mp_face_model_selection = 1
mp_face_max_num_faces = 1
mp_face_refine_landmarks = True
mp_face_min_detection_confidence = 0.5
mp_face_min_tracking_confidence = 0.5

# Mediapipe pose paramstatic_image_mode": false,
mp_pose_model_complexity = 2
mp_pose_enable_segmentation = False
mp_pose_min_detection_confidence = 0.5
mp_pose_min_tracking_confidence = 0.5

# Map from Mediapipe pose to old pose model with 17 keypoints
keypoint_mapping_table = [0, -1, 1, -1, -1, 2, -1, 3, 4, -1, -1, 5,6,7,8,9,10,-1,-1,-1,-1,-1,-1,11 ,12, 13, 14, 15, 16, -1, -1, -1, -1]


# Face feature parameters
face_detection_method = "mediapipe",
face_detection_method_choices = ["sfd", "blaze", "mediapipe", "dlib"]


min_pose_conf = 0.5,
postprocess = ["sound"],
postprocess_choices = ["sound", "face", "skeleton"]

sound_mean_norm = True
sound_windowing = True
sound_pre_emphasis = True
sound_ampl_normalization = False
sound_remove_silent = True
sound_noise_reduce = False
sound_n_mfcc = 12
sound_sample_rate = 16000
sound_window_length = 0.032
sound_window_step = 0.016
sound_feature_length = 0.512
sound_feature_step = 0.128
sound_use_pitch = True
