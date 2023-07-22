import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

'''
 Class to detect the pose from a picture using movenet-multipose. This can detect the pose of up to 6 people.   
 
 this class is based on 
 MoveNetPythonExample, by git@github.com:Kazuhito00/MoveNet-Python-Example.git
 Apache-2.0 license
'''

class PoseDetector:
    def __init__(self):
        self.input_size = 256
        model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
        module = tfhub.load(model_url)
        self.model = module.signatures['serving_default']


    def detect_pose(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # image to matching tensor
        input_image = cv.resize(image, dsize=(self.input_size, self.input_size))
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB
        input_image = input_image.reshape(-1, self.input_size, self.input_size, 3)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # create output
        outputs = self.model(input_image)

        keypoints_with_scores = outputs['output_0'].numpy()
        keypoints_with_scores = np.squeeze(keypoints_with_scores)


        keypoints_list, scores_list = [], []
        bbox_list = []
        for keypoints_with_score in keypoints_with_scores:
            keypoints = []
            scores = []
            # cast keypoints to image size
            for index in range(17):
                keypoint_x = int(image_width *
                                 keypoints_with_score[(index * 3) + 1])
                keypoint_y = int(image_height *
                                 keypoints_with_score[(index * 3) + 0])
                score = keypoints_with_score[(index * 3) + 2]

                keypoints.append([keypoint_x, keypoint_y])
                scores.append(score)

            # cast bounding box to image size
            bbox_ymin = int(image_height * keypoints_with_score[51])
            bbox_xmin = int(image_width * keypoints_with_score[52])
            bbox_ymax = int(image_height * keypoints_with_score[53])
            bbox_xmax = int(image_width * keypoints_with_score[54])
            bbox_score = keypoints_with_score[55]

            # add data to lists
            keypoints_list.append(keypoints)
            scores_list.append(scores)
            bbox_list.append(
                [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

        return keypoints_list, scores_list, bbox_list

    def draw_debug(self,
            debug_image,
            keypoints_list,
            scores_list,
            bbox_list,
            keypoint_score_th=0.2,
            bbox_score_th=0.2,

    ):

        # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
        # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
        for keypoints, scores in zip(keypoints_list, scores_list):
            # Line：鼻 → 左目
            index01, index02 = 0, 1
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：鼻 → 右目
            index01, index02 = 0, 2
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左目 → 左耳
            index01, index02 = 1, 3
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右目 → 右耳
            index01, index02 = 2, 4
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：鼻 → 左肩
            index01, index02 = 0, 5
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：鼻 → 右肩
            index01, index02 = 0, 6
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左肩 → 右肩
            index01, index02 = 5, 6
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左肩 → 左肘
            index01, index02 = 5, 7
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左肘 → 左手首
            index01, index02 = 7, 9
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右肩 → 右肘
            index01, index02 = 6, 8
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右肘 → 右手首
            index01, index02 = 8, 10
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左股関節 → 右股関節
            index01, index02 = 11, 12
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左肩 → 左股関節
            index01, index02 = 5, 11
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左股関節 → 左ひざ
            index01, index02 = 11, 13
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：左ひざ → 左足首
            index01, index02 = 13, 15
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右肩 → 右股関節
            index01, index02 = 6, 12
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右股関節 → 右ひざ
            index01, index02 = 12, 14
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)
            # Line：右ひざ → 右足首
            index01, index02 = 14, 16
            if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv.line(debug_image, point01, point02, (255, 255, 255), 4)
                cv.line(debug_image, point01, point02, (0, 0, 0), 2)

            # Circle：各点
            for keypoint, score in zip(keypoints, scores):
                if score > keypoint_score_th:
                    cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
                    cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

        # バウンディングボックス
        for bbox in bbox_list:
            if bbox[4] > bbox_score_th:
                cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             (255, 255, 255), 4)
                cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             (0, 0, 0), 2)
        '''
        # 処理時間
        cv.putText(debug_image,
                   "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
                   cv.LINE_AA)
        cv.putText(debug_image,
                   "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                   cv.LINE_AA)
        '''
        return debug_image

    def filter_keypoints_by_bbox_scores(self, keypoints_list, bbox_list, threshold):
        filtered_keypoints = []
        for i, keypoints in enumerate(keypoints_list):
            if bbox_list[i][4] > threshold:
                filtered_keypoints.append(keypoints)
        return filtered_keypoints
