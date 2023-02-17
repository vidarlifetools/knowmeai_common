import numpy as np
import pickle
import vg
from constants import mediapipe_landmarks, mediapipe_angles
import mediapipe as mp
from feature_constants import\
    mp_face_max_num_faces,\
    mp_face_refine_landmarks,\
    mp_face_min_detection_confidence,\
    mp_face_min_tracking_confidence

class ExprFeature:
    def __init__(self):
        self.config = None
        self.mediapipe_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=mp_face_max_num_faces,
            refine_landmarks=mp_face_refine_landmarks,
            min_detection_confidence=mp_face_min_detection_confidence,
            min_tracking_confidence=mp_face_min_tracking_confidence
        )
    def get(self, image):
        results = self.mediapipe_face.process(image)
        face_landmarks = np.zeros((478, 3), dtype=float)
        if results.multi_face_landmarks:
            first = True
            for landmarks in results.multi_face_landmarks:
                i=0
                if first:
                    #print(f"Length og landmarks {len(landmarks.landmark)}")
                    for landmark in landmarks.landmark:
                        face_landmarks[i, :] = [landmark.x * image.shape[1],
                                                landmark.y * image.shape[0],
                                                landmark.z * -1000.0]
                        i += 1
                    first = False
            face_landmarks = np.matmul(face_landmarks, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            return face_landmarks, True, results.multi_face_landmarks[0]
        else:
            return None, False, None

"""
class Expr(DataModule):
    name = MODULE_EXPR
    config_class = ExprConfig

    def __init__(self, config):
        super().__init__(*args)

        self.mediapipe_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:

    def view_face(self, image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        cv2.imshow('MediaPipe Expr Mesh', cv2.flip(image, 1))
        cv2.waitKey(1)
"""
class ExprPrediction:
    def __init__(self, model_filename_expression):
        with open(model_filename_expression, "rb") as file:
            self.expr_model = pickle.load(file)
        print(f"Expression model: {self.expr_model}")

    def get_feature(self, face_landmarks):
        nof_fts = len(mediapipe_angles)
        fts = np.zeros(nof_fts, dtype=np.float32)
        for i in range(nof_fts):
            vector1 = face_landmarks[mediapipe_landmarks[mediapipe_angles[i][0]], :] - face_landmarks[
                                                                                       mediapipe_landmarks[
                                                                                           mediapipe_angles[i][1]], :]
            vector2 = face_landmarks[mediapipe_landmarks[mediapipe_angles[i][2]], :] - face_landmarks[
                                                                                       mediapipe_landmarks[
                                                                                           mediapipe_angles[i][1]], :]

            if sum(np.abs(np.array(vector1))) > 0 and sum(np.abs(np.array(vector2))) > 0:
                angle = vg.angle(vector1, vector2,
                                 units='rad')  # vg.signed_angle(vector1, vector2, look=vg.basis.z, units='rad')
                # angle = angle + 2 * math.pi if angle < 0 else angle
            else:
                angle = -1.0
            fts[i] = angle

        return fts
    def get_class(self, face_landmarks):
        fts = self.get_feature(face_landmarks)
        fts = fts.reshape(1, -1)
        expr_class = self.expr_model.predict(fts)
        return expr_class

