# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from util import file
 
 
# from openpose.body.estimator import BodyPoseEstimator


# filemanager = file.FileManager()
# class OpenPose:
#     def __init__(self, filemanager):
#         self.model = BodyPoseEstimator(pretrained=True)
#         self.filemanager = filemanager

#     def predict(self, image, filename):
#         keypoints = self.model(image)
#         pose_keypoints = []

#         if len(keypoints) > 0:
#             for keypoint in keypoints[0]:
#                 pose_keypoints.append(keypoint[0].astype(float))
#                 pose_keypoints.append(keypoint[1].astype(float))
#                 pose_keypoints.append(keypoint[2].astype(float))

#             json_data = {"version": 1.0, "people": [
#                         {"person_id": [-1],
#                         "face_keypoints":[],
#                         "pose_keypoints":[pose_keypoints],
#                         "hand_right_keypoints": [], 
#                         "hand_left_keypoints":[],
#                         }]}
#             self.filemanager.save_pose(json_data, filename)
#             return "Success"
#         else:
#             self.filemanager.remove_human(filename)
#             return "Fail"
        
