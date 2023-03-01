import numpy as np
import pickle
from constants import *
import os
import json


class CompoundPrediction:
    def __init__(self, model_filename_compound, model_filename_expression, model_filename_gesture,
                 model_filename_sound):
        # Get number of classes of each sub-model
        self.nof_face_classes = get_nof_model_labels(model_filename_expression)
        self.nof_gesture_classes = get_nof_model_labels(model_filename_gesture)
        self.nof_sound_classes = get_nof_model_labels(model_filename_sound)

        # Load compound model
        self.model_filename_compound = model_filename_compound
        self.compound_model = load_model(self.model_filename_compound)

    def get_class(self, sequence_gesture_classes, sequence_expr_classes, sequence_sound_classes,
                  gesture_sequence, expr_frame_nos, sound_frame_nos, frame_sequence, current_frame, logger):

        # compute feature vector for compound recognition
        fts, fts_valid = compute_feature_vector_compound(
            expr_frame_nos, sequence_expr_classes, self.nof_face_classes,
            gesture_sequence, sequence_gesture_classes, self.nof_gesture_classes,
            sound_frame_nos, sequence_sound_classes, self.nof_sound_classes,
            frame_sequence, logger=logger, timestamp=current_frame)

        # Reshape array for model prediction
        fts = np.array(fts).reshape(1, -1)

        if not fts_valid:
            class_nos = np.array([MISSING_CLASS], dtype=np.uint8)
            class_scores = np.array([0], dtype=np.float16)
        else:
            class_nos = np.array(self.compound_model.predict(fts), dtype=np.uint8)
            class_scores = np.array(self.compound_model.decision_function(fts), dtype=np.float16).ravel()

        # Return preducted class
        return class_nos[0]


def load_model(model_filename):
    if os.path.isfile(model_filename):
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
    else:
        print(model_filename + ' does not exist')
        model = None
    return model


def get_nof_model_labels(model_filename):
    model = load_model(model_filename)
    return len(model.classes_)


# TODO expression type different name?
# TODO update docstring
def compute_one_hot_vector(sub_labels, frame_nos, sub_window, nof_classes, expression_type, logger=None,
                           timestamp=None):
    '''
    This function returns the one-hot vector for the classes associated to face expression or gesture (including the MISSING_CLASS*)

    :param nof_classes:Number of classes for expressions or gestures
    :param unique: vector of unique elements for expressions or gestures
    :param counts: vector gathering the counts of every unique element for expressions or gestures
    :param expression_type: either 'FACE' or 'GESTURE' or 'SOUND'
    :param logger: logger to write warning messages to
    :param timestamp: timestamp related to face expression (or gesture, the 2 should be the same at this point) for
    better tracking errors in timestamps in the logger
    :return: the one-hot vector of the classes
    '''
    assert (expression_type == 'FACE' or expression_type == 'GESTURE' or expression_type == 'SOUND')

    # Select all labels in sub-window
    window_frames = np.logical_and(frame_nos >= sub_window[0], frame_nos <= sub_window[1])
    sub_labels_window = sub_labels[window_frames]

    # Find number of occurences for each label
    unique, counts = np.unique(sub_labels_window, return_counts=True)

    # Initialize one-hot vector, missing class is already added to nr of classes
    one_hot_vector = [0] * (nof_classes)

    # The vector cl is transformed into a one-hot vector and the value one is assigned to the majority class
    if len(counts) != 0:
        idx = [i for i, c in enumerate(counts) if c == max(counts)][0]
        majority_class = int(unique[idx])
        if majority_class < len(one_hot_vector) - 1:
            # The majority class is different from the MISSING_CLASS*
            one_hot_vector[majority_class] = 1
        else:
            # The majority class corresponds to the MISSING_CLASS*
            if len(counts) > 1:
                # We can find a majority class when we disregard the MISSING_CLASS*
                unique = [unique[i] for i, u in enumerate(unique) if not i == idx]
                counts = [counts[i] for i, u in enumerate(counts) if not i == idx]
                idx = [i for i, c in enumerate(counts) if c == max(counts)][0]
                majority_class = int(unique[idx])
                one_hot_vector[majority_class] = 1

            else:
                # All classes are the MISSING_CLASS*, so the last element of cl (the element of the MISSING_CLASS*)
                # is assigned to 1
                one_hot_vector[-1] = 1
                if (logger is not None) and (timestamp is not None):
                    logger.debug(
                        'WARNING_' + expression_type + ': timestamp ' + str(timestamp) +
                        ' uses one hot vector with MISSING_CLASS_' \
                        + expression_type + ' as majority class')
    return one_hot_vector


def compute_feature_vector_compound(expression_frame_nos, expression_classes, nof_expr_classes,
                                    gesture_frame_sequences, gesture_classes, nof_gesture_classes,
                                    sound_frame_nos, sound_classes, nof_sound_classes,
                                    label_frame_sequences, logger=None, timestamp=None):
    '''
    This function builds a feature vector inside a window taking into account the order of the classes, the covariation
    between expression, gesture and sound labels.

    :param expression_frame_nos: vector containing the frame numbers where expression classes have been extracted
    :param expression_classes: vector containing the extracted expression classes
    :param nof_expr_classes: number of expression classes
    :param gesture_frame_sequences: vector containing the frame sequences where the gesture classes have been extracted
    :param nof_gesture_classes: number of gesture classes
    :param label_frame_sequences: array containing the frame sequence for which to compute the compound feature vector
    :param sound_frame_nos: vector containing the frame numbers where sound classes have been extracted
    :param sound_classes: vector containing the extracted sound classes
    :param nof_sound_classes: number of sound classes
    :param logger: logger where log messages are written
    :param timestamp: timestamp related to face expression (or gesture, the 2 should be the same at this point) for
    better tracking errors in timestamps in the logger
    :return: compound feature vector within a window determined by label_frame_sequences parameter
    '''

    # We consider the starting and ending frames of the labelling sliding window of size COMPOUND_BUFFER_LEN
    first_frame_no = label_frame_sequences[0]
    last_frame_no = label_frame_sequences[-1]
    assert last_frame_no - first_frame_no == COMPOUND_BUFFER_LEN, \
        print(last_frame_no - first_frame_no, "does not match", COMPOUND_BUFFER_LEN)

    # Increment nof of sub-labels by one to account for the "missing" class
    nof_expr_classes += 1
    nof_gesture_classes += 1
    nof_sound_classes += 1

    # Initialization of empty arrays for expression, gesture, sound
    expr_classes_interval, gesture_classes_interval, sound_classes_interval = \
        initialize_recognition_intervals(nof_expr_classes, nof_gesture_classes, nof_sound_classes)

    # Get the start indices of the sub-windows used to compute the one-hot vectors
    # TODO check whether it should be COMPOUND_BUFFER_LEN +1 or just COMPOUND_BUFFER_LEN
    first_inds = (np.arange(NOF_INTERVALS_COMPOUND) * (
                COMPOUND_BUFFER_LEN + 1)) / NOF_INTERVALS_COMPOUND + first_frame_no
    first_inds = first_inds.astype(int)

    # Get the end indices of the sub-windows used to compute the one-hot vectors
    last_inds = np.zeros(len(first_inds), dtype=int)
    last_inds[:-1] = first_inds[1:] - 1
    last_inds[-1] = last_frame_no

    # Initialize counts for nr of missing sub-labels
    nof_missing_face = 0.0
    nof_missing_gesture = 0.0
    nof_missing_sound = 0.0

    # Calculate one-hot vectors for each sub-window
    for i in range(NOF_INTERVALS_COMPOUND):
        # Get the sub-window
        sub_window = (first_inds[i], last_inds[i])

        # Construct the one-hot vector for each of the recognition modules
        if INCLUDE_FACE_EXPR:
            # Compute one-hot vector for sub-window
            one_hot_expr = compute_one_hot_vector(expression_classes, expression_frame_nos, sub_window,
                                                  nof_expr_classes, "FACE", logger=logger, timestamp=timestamp)

            # Fill the expression classes interval
            expr_classes_interval[(i * nof_expr_classes):((i + 1) * nof_expr_classes)] = one_hot_expr
            if one_hot_expr[-1] == 1:  # Missing face expression class for this interval
                nof_missing_face += 1.0
        else:
            expr_classes_interval = []

        if INCLUDE_GESTURE:
            # Compute one-hot vector for sub-window. For gesture, a sub-label computed based on N frames is
            # included in the window if the last of the N frames is within the window
            first_frame, last_frame = zip(*gesture_frame_sequences)
            one_hot_gesture = compute_one_hot_vector(gesture_classes, last_frame, sub_window, nof_gesture_classes,
                                                     "GESTURE", logger=logger, timestamp=timestamp)

            # Fill gesture classes interval
            gesture_classes_interval[(i * nof_gesture_classes):((i + 1) * nof_gesture_classes)] = one_hot_gesture
            if one_hot_gesture[-1] == 1:  # Missing gesture class for this interval
                nof_missing_gesture += 1.0
        else:
            gesture_classes_interval = []

        if INCLUDE_SOUND:
            # Compute one-hot vector for sub-window
            one_hot_sound = compute_one_hot_vector(sound_classes, sound_frame_nos, sub_window, nof_sound_classes,
                                                   "SOUND", logger=logger, timestamp=timestamp)

            # Fill sound classes interval
            sound_classes_interval[(i * nof_sound_classes):((i + 1) * nof_sound_classes)] = one_hot_sound
            if one_hot_sound[-1] == 1:  # Missing sound class for this interval
                nof_missing_sound += 1.0
        else:
            sound_classes_interval = []

    # Feature vector is OK if the class is missing for less than 50% of the intervals
    fts_OK = check_fts_valid(nof_missing_face, nof_missing_gesture, nof_missing_sound)

    # The compound feature vector is the concatenation of the one-hot vectors
    compound_ft_vector = expr_classes_interval + gesture_classes_interval + sound_classes_interval
    return compound_ft_vector, fts_OK


def check_fts_valid(nof_missing_face, nof_missing_gesture, nof_missing_sound):
    """ Check if compound feature vector is OK (less than 50% of any included sub-labels is missing)"""
    fts_valid = True
    if INCLUDE_FACE_EXPR and nof_missing_face / NOF_INTERVALS_COMPOUND > VALID_MISSING_PORTION:
        fts_valid = False
    if INCLUDE_GESTURE and nof_missing_gesture / NOF_INTERVALS_COMPOUND > VALID_MISSING_PORTION:
        fts_valid = False
    if INCLUDE_SOUND and nof_missing_sound / NOF_INTERVALS_COMPOUND > VALID_MISSING_PORTION:
        fts_valid = False

    return fts_valid


def initialize_recognition_intervals(nof_expr_classes, nof_gesture_classes, nof_sound_classes):
    # The features will be computed within the sliding window, by diving it into NOF_INTERVAL_COMPOUND sub-windows
    expr_classes_interval = []
    if INCLUDE_FACE_EXPR:
        expr_classes_interval = [-1] * NOF_INTERVALS_COMPOUND * nof_expr_classes

    gesture_classes_interval = []
    if INCLUDE_GESTURE:
        gesture_classes_interval = [-1] * NOF_INTERVALS_COMPOUND * nof_gesture_classes

    sound_classes_interval = []
    if INCLUDE_SOUND:
        sound_classes_interval = [-1] * NOF_INTERVALS_COMPOUND * nof_sound_classes

    return expr_classes_interval, gesture_classes_interval, sound_classes_interval


def get_sequence_labels_expr(frame_sequence, expression_classes):
    """ Get expression labels in sequence, and corresponding frame numbers """
    return get_sequence_labels(frame_sequence, expression_classes)


def get_sequence_labels_sound(frame_sequence, sound_classes):
    """ Get sound labels in sequence, and corresponding frame numbers """
    return get_sequence_labels(frame_sequence, sound_classes)


def get_sequence_labels_gesture(frame_sequence, gesture_classes):
    """ Get gesture labels in sequence, and corresponding frame sequences """
    sequence_gesture_classes, sequence_frame_numbers = get_sequence_labels(frame_sequence, gesture_classes)

    # Each gesture labels is computed based on a sequence of frames
    last_frame_gesture = sequence_frame_numbers

    # TODO check if this is sometimes negative - is that a problem?
    first_frame_gesture = sequence_frame_numbers - GESTURE_BUFFER_LEN + 1
    gesture_frame_sequences = np.array(list(zip(first_frame_gesture, last_frame_gesture)))

    return sequence_gesture_classes, gesture_frame_sequences


def get_sequence_labels(frame_sequence, labels):
    (first_frame, last_frame) = frame_sequence
    sequence_frame_numbers = np.arange(first_frame, last_frame)
    sequence_labels = np.array(labels)[first_frame:last_frame]
    return sequence_labels, sequence_frame_numbers
