import pandas
import pandas as pd
import json

class Converter:

    def dataframe_from_keypoints(self, keypoints_list) -> pandas.DataFrame:
        '''

        :param keypoints_list:
        :return:
        '''
        dicitonary = {}

        for index, keypoints in enumerate(keypoints_list):
            data = {
                'nose_x': keypoints[0][0], 'nose_y': keypoints[0][1],
                'left_eye_x': keypoints[1][0], 'left_eye_y': keypoints[1][1],
                'right_eye_x': keypoints[2][0], 'right_eye_y': keypoints[2][1],
                'left_ear_x': keypoints[3][0], 'left_ear_y': keypoints[3][1],
                'right_ear_x': keypoints[4][0], 'right_ear_y': keypoints[4][1],
                'left_shoulder_x': keypoints[5][0], 'left_shoulder_y': keypoints[5][1],
                'right_shoulder_x': keypoints[6][0], 'right_shoulder_y': keypoints[6][1],
                'left_elbow_x': keypoints[7][0], 'left_elbow_y': keypoints[7][1],
                'right_elbow_x': keypoints[8][0], 'right_elbow_y': keypoints[8][1],
                'left_wrist_x': keypoints[9][0], 'left_wrist_y': keypoints[9][1],
                'right_wrist': keypoints[10][0], 'right_wrist_y': keypoints[10][1],
                'left_hip_x': keypoints[11][0], 'left_hip_y': keypoints[11][1],
                'right_hip_x': keypoints[12][0], 'right_hip_y': keypoints[12][1],
                'left_knee_x': keypoints[13][0], 'left_knee_y': keypoints[13][1],
                'right_knee_x': keypoints[14][0], 'right_knee_y': keypoints[14][1],
                'left_ankle_x': keypoints[15][0], 'left_ankle_y': keypoints[15][1],
                'right_ankle_x': keypoints[16][0], 'right_ankle_y': keypoints[16][1]
            }
            dicitonary[index] = data
        df = pd.DataFrame.from_dict(dicitonary)
        return df

    def dataframe_from_json(self, json_data):

        return pd.DataFrame.from_dict(json.loads(json_data), orient="index")

