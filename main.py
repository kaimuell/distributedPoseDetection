import argparse

import cv2
import numpy as np

from displaySubscriber.converter import Converter
from pose_detector import PoseDetector
from publisher import Publisher

def run_camera_and_pose_detection(debug: bool):
    capture = cv2.VideoCapture(0)
    pose_detector = PoseDetector()
    publisher = Publisher()
    converter = Converter()

    while capture.isOpened():
        has_image, image = capture.read()
        if not has_image:
            raise Exception("could not read image")

        keypoints_list, scores_list, bbox_list = pose_detector.detect_pose(image)

        keypoints_list = pose_detector.filter_keypoints_by_bbox_scores(keypoints_list, bbox_list, threshold=0.2)
        json = converter.dataframe_from_keypoints(keypoints_list).to_json()
        publisher.publish(json)
        # print(converter.dataframe_from_json(json))

        if debug:
            debug_image = pose_detector.draw_debug(image, keypoints_list, scores_list, bbox_list)
            cv2.imshow("debug", debug_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    run_camera_and_pose_detection(args.debug)