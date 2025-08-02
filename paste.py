import rclpy
from rclpy.node import Node
from flask import Flask, render_template, request, jsonify, Response
import threading
import time
import sys
import traceback
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict, dataclass
import math
import logging
from collections import defaultdict, deque
from pathlib import Path

# ROS2メッセージ
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

# YOLOv8のインポート
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available. Vision system will be disabled.")
    YOLO_AVAILABLE = False

@dataclass
class PersonDetection:
    """人物検出結果"""
    person_id: str
    bbox: tuple  # x1, y1, x2, y2
    center: tuple  # center_x, center_y
    confidence: float
    direction: str
    facing_direction: str
    frame_number: int

@dataclass
class SystemConfig:
    """システム設定"""
    # カメラ設定
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # AI設定
    yolo_model_path: str = "yolov8n-pose.pt"
    confidence_threshold: float = 0.5
    
    # 追跡設定
    max_history: int = 10
    max_missing_frames: int = 30
    matching_distance_threshold: float = 100.0
    
    # 移動判定設定
    horizontal_threshold: int = 30
    depth_threshold: int = 10
    
    # ROS2通信設定
    ros2_server_url: str = "http://localhost:8080"
    http_timeout: float = 1.0
    
    # ログ設定
    log_level: str = "INFO"
    save_debug_images: bool = False
    debug_image_dir: str = "debug_images"

class PersonTracker:
    """個人の移動履歴を追跡するクラス"""
    def __init__(self, person_id: int, max_history: int = 10):
        self.person_id = person_id
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.shoulder_width_history = deque(maxlen=max_history)
        self.body_angle_history = deque(maxlen=max_history)
        self.last_seen = 0
        
    def update(self, center_pos: tuple, shoulder_width: float, body_angle: float, frame_count: int):
        """位置情報を更新"""
        self.position_history.append(center_pos)
        self.shoulder_width_history.append(shoulder_width)
        self.body_angle_history.append(body_angle)
        self.last_seen = frame_count
        
    def get_movement_direction(self) -> str:
        """移動方向を判定"""
        if len(self.position_history) < 5:
            return "unknown"
            
        recent_positions = list(self.position_history)[-7:]
        
        if len(recent_positions) < 5:
            return "unknown"
            
        x_movement = recent_positions[-1][0] - recent_positions[0][0]
        
        if len(self.shoulder_width_history) >= 3:
            recent_widths = list(self.shoulder_width_history)[-7:]
            width_change = recent_widths[-1] - recent_widths[0]
        else:
            width_change = 0
            
        horizontal_threshold = 30
        depth_threshold = 10
        
        if abs(x_movement) > horizontal_threshold:
            if x_movement > 0:
                return "moving_right"
            else:
                return "moving_left"
        elif abs(width_change) > depth_threshold:
            if width_change > 0:
                return "moving_toward"
            else:
                return "moving_away"
        else:
            return "stationary"
    
    def get_facing_direction(self, landmarks: List[tuple]) -> str:
        """YOLOv8 Poseの骨格情報から向いている方向を判定"""
        if not landmarks or len(landmarks) < 17:
            return "unknown"
            
        nose = landmarks[0]
        left_eye = landmarks[1]
        right_eye = landmarks[2]
        left_ear = landmarks[3]
        right_ear = landmarks[4]
        left_shoulder = landmarks[5]
        right_shoulder = landmarks[6]
        left_hip = landmarks[11]
        right_hip = landmarks[12]
        
        VISIBLE_THRESHOLD = 0.5
        
        # 顔パーツの可視性チェック
        face_parts_visible = {
            'nose': nose[2] > VISIBLE_THRESHOLD,
            'left_eye': left_eye[2] > VISIBLE_THRESHOLD,
            'right_eye': right_eye[2] > VISIBLE_THRESHOLD,
            'left_ear': left_ear[2] > VISIBLE_THRESHOLD,
            'right_ear': right_ear[2] > VISIBLE_THRESHOLD
        }
        
        num_face_visible = sum(1 for v in face_parts_visible.values() if v)
        
        # 体幹パーツの可視性チェック
        torso_parts_visible = {
            'left_shoulder': left_shoulder[2] > VISIBLE_THRESHOLD,
            'right_shoulder': right_shoulder[2] > VISIBLE_THRESHOLD,
            'left_hip': left_hip[2] > VISIBLE_THRESHOLD,
            'right_hip': right_hip[2] > VISIBLE_THRESHOLD
        }
        num_torso_visible = sum(1 for v in torso_parts_visible.values() if v)

        # 後ろ向きの判定
        if num_face_visible <= 1 and num_torso_visible >= 2:
            if torso_parts_visible['left_shoulder'] and torso_parts_visible['right_shoulder']:
                if left_shoulder[0] > right_shoulder[0] + 15:
                    return "back_right"
                elif right_shoulder[0] > left_shoulder[0] + 15:
                    return "back_left"
                else:
                    return "back"
            return "back"

        # 正面・左右向きの判定
        if num_face_visible >= 3:
            eye_center_x = None
            if face_parts_visible['left_eye'] and face_parts_visible['right_eye']:
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
            elif face_parts_visible['left_eye']:
                eye_center_x = left_eye[0]
            elif face_parts_visible['right_eye']:
                eye_center_x = right_eye[0]

            if face_parts_visible['nose'] and eye_center_x is not None:
                nose_offset_from_eye_center = nose[0] - eye_center_x
                
                if nose_offset_from_eye_center < -15:
                    return "left"
                elif nose_offset_from_eye_center > 15:
                    return "right"
                else:
                    return "front"
            
            if face_parts_visible['left_ear'] and not face_parts_visible['right_ear']:
                return "left"
            if face_parts_visible['right_ear'] and not face_parts_visible['left_ear']:
                return "right"

            return "front"
        
        return "unknown"

class PersonDetectionSystem:
    """YOLOv8 Poseを使用した複数人の移動方向検出システム"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # YOLOv8 Poseモデル初期化
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.logger.info(f"YOLOv8 Poseモデル読み込み中: {config.yolo_model_path}")
                self.yolo = YOLO(config.yolo_model_path)
                self.logger.info("YOLOv8 Poseモデル読み込み完了")
            except Exception as e:
                self.logger.error(f"YOLO model loading failed: {e}")
                self.yolo = None
        
        # 追跡システム
        self.person_trackers: Dict[int, PersonTracker] = {}
        self.next_person_id = 0
        self.frame_count = 0
        self.person_poses: Dict[int, List] = {}
        self.person_facing_directions: Dict[int, str] = {}
        
        # カメラ初期化
        self.cap = None
        self.initialize_camera()
        
        # デバッグ画像保存用
        if config.save_debug_images:
            Path(config.debug_image_dir).mkdir(exist_ok=True)
    
    def initialize_camera(self) -> bool:
        """カメラ初期化"""
        try:
            self.cap = cv2.VideoCapture(self.config.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"カメラ {self.config.camera_id} を開けません")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            self.logger.info(f"カメラ初期化完了: {self.config.camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            return False
    
    def extract_pose_features(self, keypoints: np.ndarray, bbox: List[int]) -> Optional[tuple]:
        """YOLOv8 Poseのキーポイントから特徴を抽出"""
        if keypoints is None or len(keypoints) == 0:
            return None
            
        landmarks = []
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints):
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                landmarks.append((x, y, conf))
        
        if len(landmarks) < 17:
            return None
            
        left_shoulder = landmarks[5]
        right_shoulder = landmarks[6]
        left_hip = landmarks[11]
        right_hip = landmarks[12]
        
        if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or
            left_hip[2] < 0.3 or right_hip[2] < 0.3):
            return None
            
        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
        center_pos = (center_x, center_y)
        
        shoulder_width = math.sqrt(
            (right_shoulder[0] - left_shoulder[0])**2 + 
            (right_shoulder[1] - left_shoulder[1])**2
        )
        
        shoulder_angle = math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ) * 180 / math.pi
        
        return center_pos, shoulder_width, shoulder_angle, landmarks
    
    def update_trackers(self, detections):
        """検出結果を用いてトラッカーを更新"""
        self.frame_count += 1
        
        if not detections:
            for tracker_id, tracker in list(self.person_trackers.items()):
                if self.frame_count - tracker.last_seen > self.config.max_missing_frames:
                    if tracker_id in self.person_poses:
                        del self.person_poses[tracker_id]
                    if tracker_id in self.person_facing_directions:
                        del self.person_facing_directions[tracker_id]
                    del self.person_trackers[tracker_id]
            return
            
        used_detections = set()
        
        for tracker_id, tracker in list(self.person_trackers.items()):
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, (bbox, pose_features) in enumerate(detections):
                if i in used_detections:
                    continue
                    
                if tracker.position_history:
                    last_pos = tracker.position_history[-1]
                    center_pos = pose_features[0]
                    distance = math.sqrt((center_pos[0] - last_pos[0])**2 + (center_pos[1] - last_pos[1])**2)
                    
                    if distance < self.config.matching_distance_threshold and distance < best_distance:
                        best_match = pose_features
                        best_distance = distance
                        best_idx = i
                        
            if best_match:
                center_pos, shoulder_width, body_angle, landmarks = best_match
                tracker.update(center_pos, shoulder_width, body_angle, self.frame_count)
                self.person_poses[tracker_id] = landmarks
                facing_direction = tracker.get_facing_direction(landmarks)
                self.person_facing_directions[tracker_id] = facing_direction
                used_detections.add(best_idx)
            else:
                if self.frame_count - tracker.last_seen > self.config.max_missing_frames:
                    if tracker_id in self.person_poses:
                        del self.person_poses[tracker_id]
                    if tracker_id in self.person_facing_directions:
                        del self.person_facing_directions[tracker_id]
                    del self.person_trackers[tracker_id]
                    
        for i, (bbox, pose_features) in enumerate(detections):
            if i not in used_detections:
                new_tracker = PersonTracker(self.next_person_id, self.config.max_history)
                center_pos, shoulder_width, body_angle, landmarks = pose_features
                new_tracker.update(center_pos, shoulder_width, body_angle, self.frame_count)
                self.person_trackers[self.next_person_id] = new_tracker
                self.person_poses[self.next_person_id] = landmarks
                facing_direction = new_tracker.get_facing_direction(landmarks)
                self.person_facing_directions[self.next_person_id] = facing_direction
                self.next_person_id += 1
    
    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """フレームから人物を検出"""
        if self.yolo is None:
            return []
        
        try:
            results = self.yolo(frame)
            detections = []
            
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and keypoints is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        kpts = keypoints[i]
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > self.config.confidence_threshold:
                            if hasattr(kpts, 'data') and kpts.data is not None and len(kpts.data.shape) >= 2:
                                if len(kpts.data.shape) == 3:
                                    keypoint_data = kpts.data[0].cpu().numpy()
                                else:
                                    keypoint_data = kpts.data.cpu().numpy()
                                    
                                keypoint_flat = keypoint_data.flatten()
                                
                                pose_features = self.extract_pose_features(keypoint_flat, [x1, y1, x2, y2])
                                if pose_features:
                                    detections.append(([x1, y1, x2, y2], pose_features))
            
            self.update_trackers(detections)
            
            # PersonDetectionオブジェクトのリストを生成
            person_detections = []
            for tracker_id, tracker in self.person_trackers.items():
                if tracker.position_history:
                    center = tracker.position_history[-1]
                    movement_direction = tracker.get_movement_direction()
                    facing_direction = self.person_facing_directions.get(tracker_id, "unknown")
                    
                    # バウンディングボックスを計算
                    pose_landmarks = self.person_poses.get(tracker_id)
                    if pose_landmarks:
                        try:
                            keypoints_for_bbox = [pose_landmarks[5], pose_landmarks[6], 
                                                  pose_landmarks[11], pose_landmarks[12]]
                            
                            valid_kpts = [(kp[0], kp[1]) for kp in keypoints_for_bbox if kp[2] > 0.3]
                            
                            if valid_kpts:
                                min_x = min(kp[0] for kp in valid_kpts)
                                max_x = max(kp[0] for kp in valid_kpts)
                                min_y = min(kp[1] for kp in valid_kpts)
                                max_y = max(kp[1] for kp in valid_kpts)
                                
                                padding_x = (max_x - min_x) * 0.1
                                padding_y = (max_y - min_y) * 0.05
                                bbox = (
                                    max(0, min_x - padding_x),
                                    max(0, min_y - padding_y * 2),
                                    min(frame.shape[1], max_x + padding_x),
                                    min(frame.shape[0], max_y + padding_y)
                                )
                            else:
                                bbox = (center[0] - 50, center[1] - 50, center[0] + 50, center[1] + 50)
                        except:
                            bbox = (center[0] - 50, center[1] - 50, center[0] + 50, center[1] + 50)
                    else:
                        bbox = (center[0] - 50, center[1] - 50, center[0] + 50, center[1] + 50)
                    
                    detection = PersonDetection(
                        person_id=f"person_{tracker_id}",
                        bbox=bbox,
                        center=center,
                        confidence=0.8,
                        direction=movement_direction,
                        facing_direction=facing_direction,
                        frame_number=self.frame_count
                    )
                    person_detections.append(detection)
            
            return person_detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def draw_pose_landmarks(self, image: np.ndarray, landmarks: List[tuple], person_id: int, color: tuple):
        """YOLOv8 Poseの骨格情報を描画"""
        if not landmarks or len(landmarks) < 17:
            return
            
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for i, (x, y, visibility) in enumerate(landmarks):
            if visibility > 0.5:
                cv2.circle(image, (int(x), int(y)), 4, color, -1)
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                if start_landmark[2] > 0.5 and end_landmark[2] > 0.5:
                    start_point = (int(start_landmark[0]), int(start_landmark[1]))
                    end_point = (int(end_landmark[0]), int(end_landmark[1]))
                    cv2.line(image, start_point, end_point, color, 2)
    
    def draw_detections(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """検出結果を画像に描画"""
        annotated_frame = frame.copy()
        
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for detection in detections:
            try:
                person_id = int(detection.person_id.split('_')[1])
                color = colors[person_id % len(colors)]
                
                x1, y1, x2, y2 = detection.bbox
                center_x, center_y = detection.center
                
                # バウンディングボックス描画
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 中心点描画
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                
                # ラベル作成
                direction_map = {
                    "moving_right": "Right", "moving_left": "Left", "moving_toward": "Forward", 
                    "moving_away": "Back", "stationary": "Still", "unknown": "Unknown"
                }
                
                facing_map = {
                    "right": "Right", "left": "Left", "front": "Front", 
                    "back": "Back", "back_right": "Back-R", "back_left": "Back-L",
                    "unknown": "Unknown"
                }
                
                movement_text = direction_map.get(detection.direction, detection.direction)
                facing_text = facing_map.get(detection.facing_direction, detection.facing_direction)
                
                label = f"ID:{person_id} Move:{movement_text} Face:{facing_text}"
                
                # テキスト描画
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 骨格描画
                if person_id in self.person_poses:
                    self.draw_pose_landmarks(annotated_frame, self.person_poses[person_id], person_id, color)
                
                # 軌跡描画
                if person_id in self.person_trackers:
                    tracker = self.person_trackers[person_id]
                    if len(tracker.position_history) > 1:
                        points = [(int(pos[0]), int(pos[1])) for pos in tracker.position_history]
                        for i in range(1, len(points)):
                            cv2.line(annotated_frame, points[i-1], points[i], (255, 255, 0), 2)
                
            except Exception as e:
                self.logger.error(f"Drawing error for detection {detection.person_id}: {e}")
                continue
        
        # フレーム情報
        frame_info = f"Frame: {self.frame_count}, Persons: {len(detections)}"
        cv2.putText(annotated_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def cleanup(self) -> None:
        """リソース解放"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("システム終了")

class CameraManager:
    """カメラ管理クラス"""
    
    def __init__(self):
        self.available_cameras = {}
        self.current_camera_id = 0
        self.cap = None
        self.lock = threading.Lock()
        self.scan_cameras()
    
    def scan_cameras(self):
        """利用可能なカメラをスキャン"""
        self.available_cameras = {}
        
        # 最大10台のカメラをチェック
        for camera_id in range(10):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # カメラの基本情報を取得
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # カメラ名を取得
                camera_name = f"Camera {camera_id}"
                try:
                    backend = cap.getBackendName()
                    camera_name = f"Camera {camera_id} ({backend})"
                except:
                    pass
                
                self.available_cameras[camera_id] = {
                    'id': camera_id,
                    'name': camera_name,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'status': 'available'
                }
                cap.release()
            
        print(f"Found {len(self.available_cameras)} cameras: {list(self.available_cameras.keys())}")
    
    def switch_camera(self, camera_id: int) -> bool:
        """カメラを切り替え"""
        with self.lock:
            if camera_id not in self.available_cameras:
                return False
            
            # 現在のカメラを解放
            if self.cap:
                self.cap.release()
            
            # 新しいカメラを初期化
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                # カメラ設定を適用
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.current_camera_id = camera_id
                
                # カメラ状態を更新
                for cam_id in self.available_cameras:
                    self.available_cameras[cam_id]['status'] = 'available'
                self.available_cameras[camera_id]['status'] = 'active'
                
                return True
            else:
                return False
    
    def get_current_camera(self):
        """現在のカメラ情報を取得"""
        return self.available_cameras.get(self.current_camera_id, {})
    
    def get_available_cameras(self):
        """利用可能なカメラ一覧を取得"""
        return self.available_cameras
    
    def capture_frame(self):
        """フレームを取得"""
        with self.lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return ret, frame
            return False, None
    
    def release(self):
        """カメラを解放"""
        with self.lock:
            if self.cap:
                self.cap.release()

class ROS2Interface(Node):
    """ROS2との通信を担当するクラス"""
    
    def __init__(self):
        super().__init__('flask_robot_controller')
        
        # Publisher設定
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.command_pub = self.create_publisher(String, '/robot_command', 10)
        
        # Subscriber設定
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        # ロボットの状態を保存
        self.robot_status = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'velocity': {'linear': 0, 'angular': 0},
            'laser_data': {'min_distance': 0, 'ranges_count': 0},
            'last_update': time.time()
        }
        
    def laser_callback(self, msg):
        """レーザーセンサーデータのコールバック"""
        try:
            if msg.ranges:
                valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
                if valid_ranges:
                    min_distance = min(valid_ranges)
                    self.robot_status['laser_data'] = {
                        'min_distance': min_distance,
                        'ranges_count': len(msg.ranges)
                    }
                    self.robot_status['last_update'] = time.time()
        except Exception as e:
            self.get_logger().error(f'Laser callback error: {e}')
    
    def move_robot(self, linear_x, angular_z):
        """ロボットを移動させる"""
        try:
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_vel_pub.publish(twist)
            
            # 状態更新
            self.robot_status['velocity']['linear'] = linear_x
            self.robot_status['velocity']['angular'] = angular_z
            
            return True
        except Exception as e:
            self.get_logger().error(f'Move robot error: {e}')
            return False
    
    def stop_robot(self):
        """ロボットを停止"""
        return self.move_robot(0.0, 0.0)
    
    def send_custom_command(self, command):
        """カスタムコマンドを送信"""
        try:
            msg = String()
            msg.data = command
            self.command_pub.publish(msg)
            self.get_logger().info(f'Command sent: {command}')
            return True
        except Exception as e:
            self.get_logger().error(f'Send command error: {e}')
            return False
    
    def get_robot_status(self):
        """ロボットの状態を取得"""
        return self.robot_status.copy()

class VisionSystemManager:
    """人物検出システム管理クラス（カメラ切り替え対応）"""
    
    def __init__(self):
        # カメラマネージャー初期化
        self.camera_manager = CameraManager()
        
        # 人物検出システムの設定
        self.vision_config = SystemConfig()
        self.vision_config.camera_id = 0
        self.vision_config.save_debug_images = False
        
        # 検出システム初期化
        try:
            # カスタムカメラマネージャーを使用するため、PersonDetectionSystemは直接カメラを使わない
            self.vision_system = PersonDetectionSystem(self.vision_config)
            # PersonDetectionSystemのカメラを無効化
            if hasattr(self.vision_system, 'cap') and self.vision_system.cap:
                self.vision_system.cap.release()
            self.vision_system.cap = None
            
            self.vision_active = True
        except Exception as e:
            print(f"Vision system initialization failed: {e}")
            self.vision_system = None
            self.vision_active = False
        
        # 検出結果の保存
        self.latest_detections: List[PersonDetection] = []
        self.detection_history: List[Dict] = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 統計情報
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'fps': 0.0
        }
        
        # 初期カメラ設定
        self.camera_manager.switch_camera(0)
    
    def start_vision_thread(self):
        """人物検出スレッドを開始"""
        if not self.vision_active:
            return False
        
        vision_thread = threading.Thread(
            target=self._vision_loop,
            daemon=True
        )
        vision_thread.start()
        return True
    
    def switch_camera(self, camera_id: int) -> bool:
        """カメラを切り替え"""
        try:
            success = self.camera_manager.switch_camera(camera_id)
            if success:
                print(f"Camera switched to {camera_id}")
                return True
            else:
                print(f"Failed to switch to camera {camera_id}")
                return False
        except Exception as e:
            print(f"Camera switch error: {e}")
            return False
    
    def get_available_cameras(self):
        """利用可能なカメラ一覧を取得"""
        return self.camera_manager.get_available_cameras()
    
    def get_current_camera(self):
        """現在のカメラ情報を取得"""
        return self.camera_manager.get_current_camera()
    
    def _vision_loop(self):
        """人物検出ループ"""
        if not self.vision_system:
            return
        
        fps_counter = 0
        last_fps_time = time.time()
        
        try:
            while self.vision_active:
                # カメラマネージャーからフレームを取得
                ret, frame = self.camera_manager.capture_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # 人物検出実行
                detections = self.vision_system.detect_persons(frame)
                
                # 結果を描画
                annotated_frame = self.vision_system.draw_detections(frame, detections)
                
                # 現在のカメラ情報を画面に表示
                current_camera = self.get_current_camera()
                camera_info = f"Camera: {current_camera.get('name', 'Unknown')}"
                cv2.putText(annotated_frame, camera_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 結果を保存
                with self.frame_lock:
                    self.latest_detections = detections
                    self.latest_frame = annotated_frame
                    
                    # 統計更新
                    self.stats['total_frames'] += 1
                    self.stats['total_detections'] += len(detections)
                    
                    # 検出履歴に追加
                    if detections:
                        detection_record = {
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': self.stats['total_frames'],
                            'camera_id': self.camera_manager.current_camera_id,
                            'detections': [
                                {
                                    'person_id': d.person_id,
                                    'direction': d.direction,
                                    'confidence': d.confidence,
                                    'center': d.center
                                }
                                for d in detections
                            ]
                        }
                        self.detection_history.append(detection_record)
                        
                        # 履歴サイズ制限
                        if len(self.detection_history) > 100:
                            self.detection_history.pop(0)
                
                # FPS計算
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    if elapsed > 0:
                        self.stats['fps'] = 30 / elapsed
                    last_fps_time = current_time
                
                time.sleep(0.03)  # 約30FPS
                
        except Exception as e:
            print(f"Vision loop error: {e}")
    
    def get_latest_detections(self):
        """最新の検出結果を取得"""
        with self.frame_lock:
            return self.latest_detections.copy()
    
    def get_detection_history(self):
        """検出履歴を取得"""
        with self.frame_lock:
            return self.detection_history.copy()
    
    def get_latest_frame(self):
        """最新のフレームを取得"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_stats(self):
        """統計情報を取得"""
        current_stats = self.stats.copy()
        current_stats['uptime'] = time.time() - self.stats['start_time']
        current_stats['current_camera'] = self.get_current_camera()
        return current_stats
    
    def cleanup(self):
        """リソース解放"""
        self.vision_active = False
        self.camera_manager.release()

# Flaskアプリケーションの作成
app = Flask(__name__)
app.config['DEBUG'] = True


