import numpy as np
import pickle
import vg
from constants import mediapipe_landmarks, mediapipe_angles

class expr_prediction:
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

