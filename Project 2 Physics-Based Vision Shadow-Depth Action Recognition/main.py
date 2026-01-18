import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from scipy import ndimage

class PhysicsBasedShadowAnalyzer:
    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )

        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.light_source_position = None
        self.light_intensity = 1.0
        self.shadow_sharpness_factor = 0.0

        self.shadow_threshold_dark = 25
        self.shadow_threshold_adaptive = True
        self.penumbra_threshold = 15

        self.distance_threshold_cm = 2.0  

        self.eating_threshold_cm = 3.0
        self.touching_threshold_cm = 2.0

        self.focal_length_px = 600  

        self.hand_real_width_cm = 8.0  

        self.face_real_width_cm = 15.0  

        self.action_history = deque(maxlen=15)
        self.distance_history = deque(maxlen=20)
        self.shadow_area_history = deque(maxlen=10)
        self.current_action = "No Action"
        self.action_confidence = 0.0

        self.matrix_size = (256, 256)
        self.intensity_matrix = np.zeros(self.matrix_size, dtype=np.float32)

        self.frame_times = deque(maxlen=30)

    def estimate_light_source_3d(self, frame):
        """Advanced 3D light source estimation using gradient analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        grid_h, grid_w = h // 3, w // 3
        brightness_map = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                region = blurred[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                brightness_map[i, j] = np.mean(region)

        max_idx = np.unravel_index(np.argmax(brightness_map), brightness_map.shape)

        angle_map = {
            (0, 0): 135, (0, 1): 90, (0, 2): 45,
            (1, 0): 180, (1, 1): 0, (1, 2): 0,
            (2, 0): 225, (2, 1): 270, (2, 2): 315
        }

        angle = angle_map.get(max_idx, 0)

        self.light_intensity = np.max(brightness_map) / 255.0

        x_pos = (max_idx[1] - 1) / 2.0  

        y_pos = (max_idx[0] - 1) / 2.0  

        z_pos = self.light_intensity  

        self.light_source_position = (x_pos, y_pos, z_pos, angle)

        return self.light_source_position

    def extract_hand_mask_advanced(self, frame, hand_landmarks):
        """Create detailed hand mask with proper contours"""
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        points = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = np.mean(points, axis=0).astype(int)

        x, y, hw, hh = cv2.boundingRect(hull)
        hand_width_px = max(hw, hh)

        return mask, points, (cx, cy), hand_width_px

    def extract_face_region_advanced(self, frame, face_landmarks):
        """Extract face region with mouth detection"""
        h, w, _ = frame.shape

        face_points = []
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            face_points.append([x, y])

        face_points = np.array(face_points, dtype=np.int32)

        hull = cv2.convexHull(face_points)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)

        M = cv2.moments(face_mask)
        if M['m00'] != 0:
            face_cx = int(M['m10'] / M['m00'])
            face_cy = int(M['m01'] / M['m00'])
        else:
            face_cx, face_cy = np.mean(face_points, axis=0).astype(int)

        mouth_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        mouth_points = []

        for idx in mouth_landmarks:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                mouth_points.append([x, y])

        mouth_points = np.array(mouth_points, dtype=np.int32) if mouth_points else face_points[:5]
        mouth_center = np.mean(mouth_points, axis=0).astype(int)

        fx, fy, fw, fh = cv2.boundingRect(hull)
        face_width_px = max(fw, fh)

        return face_mask, face_points, (face_cx, face_cy), mouth_center, face_width_px

    def detect_shadow_physics_based(self, frame, hand_mask, face_mask):
        """Advanced shadow detection using physical principles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        combined_mask = cv2.bitwise_or(hand_mask, face_mask)
        reference_region = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(combined_mask))

        if np.sum(reference_region > 0) > 100:
            reference_brightness = np.percentile(reference_region[reference_region > 0], 70)
        else:
            reference_brightness = np.mean(gray)

        if self.shadow_threshold_adaptive:
            threshold = max(15, int(reference_brightness * 0.15))
        else:
            threshold = self.shadow_threshold_dark

        umbra_threshold = reference_brightness - threshold
        umbra_mask = (gray < umbra_threshold).astype(np.uint8) * 255

        penumbra_threshold = reference_brightness - self.penumbra_threshold
        penumbra_mask = ((gray < penumbra_threshold) & (gray >= umbra_threshold)).astype(np.uint8) * 255

        full_shadow = cv2.bitwise_or(umbra_mask, penumbra_mask)

        shadow_on_face = cv2.bitwise_and(full_shadow, full_shadow, mask=face_mask)

        shadow_on_face = cv2.bitwise_and(shadow_on_face, shadow_on_face, mask=cv2.bitwise_not(hand_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shadow_on_face = cv2.morphologyEx(shadow_on_face, cv2.MORPH_OPEN, kernel)
        shadow_on_face = cv2.morphologyEx(shadow_on_face, cv2.MORPH_CLOSE, kernel)

        if np.sum(shadow_on_face > 0) > 50:
            edges = cv2.Canny(shadow_on_face, 50, 150)
            edge_density = np.sum(edges > 0) / max(1, np.sum(shadow_on_face > 0))
            self.shadow_sharpness_factor = np.clip(edge_density * 10, 0, 1)
        else:
            self.shadow_sharpness_factor = 0.0

        return shadow_on_face, umbra_mask, penumbra_mask

    def calculate_depth_using_physics(self, shadow_area, hand_width_px, face_width_px, 
                                     hand_centroid, mouth_center, shadow_sharpness):
        """
        Advanced depth calculation using multiple physics principles:
        1. Inverse Square Law: I ∝ 1/r²
        2. Shadow Projection Geometry
        3. Similar Triangles (pinhole camera model)
        4. Shadow Sharpness (penumbra analysis)
        """

        if hand_width_px > 0 and face_width_px > 0:

            hand_depth_cm = (self.focal_length_px * self.hand_real_width_cm) / hand_width_px
            face_depth_cm = (self.focal_length_px * self.face_real_width_cm) / face_width_px
            geometric_distance = abs(hand_depth_cm - face_depth_cm)
        else:
            geometric_distance = 50.0  

        self.shadow_area_history.append(shadow_area)
        avg_shadow_area = np.mean(list(self.shadow_area_history))

        if avg_shadow_area > 100:

            shadow_factor = np.sqrt(avg_shadow_area / 1000.0)
            shadow_depth_cm = 20.0 / max(0.1, shadow_factor)
        else:
            shadow_depth_cm = geometric_distance

        euclidean_2d = np.linalg.norm(np.array(hand_centroid) - np.array(mouth_center))

        if face_width_px > 0:
            px_to_cm = self.face_real_width_cm / face_width_px
            euclidean_cm = euclidean_2d * px_to_cm
        else:
            euclidean_cm = euclidean_2d * 0.1

        sharpness_modifier = 1.0 - (shadow_sharpness * 0.5)

        if avg_shadow_area > 500:

            weights = [0.2, 0.5, 0.2, 0.1]
            combined = (
                weights[0] * geometric_distance +
                weights[1] * shadow_depth_cm +
                weights[2] * euclidean_cm +
                weights[3] * (euclidean_cm * sharpness_modifier)
            )
        else:

            weights = [0.5, 0.1, 0.3, 0.1]
            combined = (
                weights[0] * geometric_distance +
                weights[1] * shadow_depth_cm +
                weights[2] * euclidean_cm +
                weights[3] * (euclidean_cm * sharpness_modifier)
            )

        self.distance_history.append(combined)
        smoothed_distance = np.median(list(self.distance_history))

        final_distance = np.clip(smoothed_distance, 0.2, 60.0)

        return final_distance, geometric_distance, shadow_depth_cm, euclidean_cm

    def create_intensity_matrix_advanced(self, frame, shadow_mask, face_mask, face_points):
        """Generate detailed intensity matrix with multiple layers"""
        h, w = shadow_mask.shape

        x, y, fw, fh = cv2.boundingRect(face_points)

        if fw > 10 and fh > 10:

            face_region = shadow_mask[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]
            gray_region = cv2.cvtColor(frame[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)], 
                                      cv2.COLOR_BGR2GRAY)

            shadow_layer = cv2.resize(face_region, self.matrix_size, interpolation=cv2.INTER_LINEAR)

            brightness_layer = cv2.resize(gray_region, self.matrix_size, interpolation=cv2.INTER_LINEAR)

            light_loss = 255 - brightness_layer

            combined = (shadow_layer.astype(np.float32) * 0.6 + 
                       light_loss.astype(np.float32) * 0.4)

            combined = cv2.GaussianBlur(combined, (5, 5), 0)

            alpha = 0.3
            self.intensity_matrix = alpha * combined + (1 - alpha) * self.intensity_matrix

            matrix_normalized = np.clip(self.intensity_matrix, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(matrix_normalized, cv2.COLORMAP_JET)

            grid_step = self.matrix_size[0] // 8
            for i in range(0, self.matrix_size[0], grid_step):
                cv2.line(heatmap, (i, 0), (i, self.matrix_size[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(heatmap, (0, i), (self.matrix_size[0], i), (255, 255, 255), 1, cv2.LINE_AA)

            scale_height = 20
            scale = np.linspace(0, 255, self.matrix_size[0]).reshape(1, -1)
            scale = np.repeat(scale, scale_height, axis=0).astype(np.uint8)
            scale_colored = cv2.applyColorMap(scale, cv2.COLORMAP_JET)

            heatmap_with_scale = np.vstack([heatmap, scale_colored])

        else:
            heatmap_with_scale = np.zeros((self.matrix_size[0] + 20, self.matrix_size[1], 3), dtype=np.uint8)

        return heatmap_with_scale

    def classify_action_advanced(self, distance_cm, shadow_area, hand_to_mouth_dist, velocity):
        """
        Advanced action classification with multiple criteria:
        - Distance threshold
        - Shadow area
        - Hand trajectory
        - Movement velocity
        """

        distance_score = max(0, 1 - (distance_cm / 10.0))
        shadow_score = min(1, shadow_area / 5000.0)
        proximity_score = max(0, 1 - (hand_to_mouth_dist / 100.0))

        overall_confidence = (distance_score * 0.5 + shadow_score * 0.3 + proximity_score * 0.2)

        if distance_cm <= self.touching_threshold_cm and overall_confidence > 0.5:
            action = "TOUCHING FACE"
            color = (0, 0, 255)  

        elif distance_cm <= self.eating_threshold_cm and overall_confidence > 0.4 and shadow_area > 1000:
            action = "EATING / DRINKING"
            color = (0, 100, 255)  

        elif distance_cm <= 5.0:
            action = "Hand Near Face"
            color = (0, 255, 255)  

        elif distance_cm <= 10.0:
            action = "Hand Approaching"
            color = (255, 255, 0)  

        else:
            action = "No Action"
            color = (0, 255, 0)  

        self.action_history.append((action, overall_confidence))

        if len(self.action_history) >= 8:

            recent_actions = [a[0] for a in list(self.action_history)[-8:]]
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            most_common = max(action_counts, key=action_counts.get)

            if action_counts[most_common] >= 5:
                self.current_action = most_common
                self.action_confidence = overall_confidence
            else:
                self.current_action = action
                self.action_confidence = overall_confidence
        else:
            self.current_action = action
            self.action_confidence = overall_confidence

        return self.current_action, self.action_confidence, color

    def process_frame(self, frame):
        """Main processing pipeline with enhanced physics"""
        start_time = time.time()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        light_info = self.estimate_light_source_3d(frame)

        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        distance_cm = None
        distance_mm = None
        metrics = {}
        heatmap = np.zeros((self.matrix_size[0] + 20, self.matrix_size[1], 3), dtype=np.uint8)

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            face_landmarks = face_results.multi_face_landmarks[0]

            hand_mask, hand_points, hand_centroid, hand_width_px = self.extract_hand_mask_advanced(
                frame, hand_landmarks)

            face_mask, face_points, face_centroid, mouth_center, face_width_px = self.extract_face_region_advanced(
                frame, face_landmarks)

            shadow_mask, umbra, penumbra = self.detect_shadow_physics_based(
                frame, hand_mask, face_mask)

            shadow_area = np.sum(shadow_mask > 0)

            hand_to_mouth_dist = np.linalg.norm(
                np.array(hand_centroid) - np.array(mouth_center))

            velocity = 0.0
            if len(self.distance_history) > 2:
                velocity = abs(self.distance_history[-1] - self.distance_history[-2])

            distance_cm, geo_dist, shadow_dist, eucl_dist = self.calculate_depth_using_physics(
                shadow_area, hand_width_px, face_width_px, 
                hand_centroid, mouth_center, self.shadow_sharpness_factor
            )

            distance_mm = distance_cm * 10  

            action, confidence, action_color = self.classify_action_advanced(
                distance_cm, shadow_area, hand_to_mouth_dist, velocity
            )

            heatmap = self.create_intensity_matrix_advanced(
                frame, shadow_mask, face_mask, face_points)

            metrics = {
                'distance_cm': distance_cm,
                'distance_mm': distance_mm,
                'shadow_area': shadow_area,
                'shadow_sharpness': self.shadow_sharpness_factor,
                'light_angle': light_info[3] if light_info else 0,
                'light_intensity': self.light_intensity,
                'action': action,
                'confidence': confidence,
                'action_color': action_color,
                'geometric_dist': geo_dist,
                'shadow_dist': shadow_dist,
                'euclidean_dist': eucl_dist,
                'hand_to_mouth': hand_to_mouth_dist,
                'velocity': velocity
            }

            self.draw_advanced_visualizations(
                frame, hand_points, face_points, mouth_center,
                shadow_mask, umbra, penumbra, metrics
            )

        process_time = time.time() - start_time
        self.frame_times.append(process_time)
        fps = 1.0 / np.mean(list(self.frame_times)) if self.frame_times else 0

        metrics['fps'] = fps

        return frame, heatmap, metrics

    def draw_advanced_visualizations(self, frame, hand_points, face_points, mouth_center,
                                    shadow_mask, umbra, penumbra, metrics):
        """Enhanced visualization with all information"""
        h, w, _ = frame.shape

        hand_overlay = frame.copy()
        cv2.polylines(hand_overlay, [hand_points], True, (0, 255, 0), 2)
        cv2.fillConvexPoly(hand_overlay, hand_points, (0, 255, 0))
        cv2.addWeighted(hand_overlay, 0.1, frame, 0.9, 0, frame)

        hand_centroid = np.mean(hand_points, axis=0).astype(int)
        cv2.circle(frame, tuple(hand_centroid), 7, (0, 255, 0), -1)
        cv2.circle(frame, tuple(hand_centroid), 9, (255, 255, 255), 2)

        face_hull = cv2.convexHull(face_points)
        face_overlay = frame.copy()
        cv2.polylines(face_overlay, [face_hull], True, (255, 0, 0), 2)
        cv2.fillConvexPoly(face_overlay, face_hull, (255, 0, 0))
        cv2.addWeighted(face_overlay, 0.1, frame, 0.9, 0, frame)

        cv2.circle(frame, tuple(mouth_center), 7, (255, 0, 255), -1)
        cv2.circle(frame, tuple(mouth_center), 9, (255, 255, 255), 2)

        shadow_overlay = np.zeros_like(frame)
        shadow_overlay[:, :, 2] = umbra  

        shadow_overlay[:, :, 0] = penumbra  

        cv2.addWeighted(frame, 1.0, shadow_overlay, 0.4, 0, frame)

        cv2.line(frame, tuple(hand_centroid), tuple(mouth_center), 
                (255, 255, 0), 3, cv2.LINE_AA)

        panel_height = 280
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        y_offset = 25
        line_height = 25

        distance_cm = metrics.get('distance_cm', 0)
        distance_mm = metrics.get('distance_mm', 0)
        cv2.putText(panel, f"DISTANCE: {distance_cm:.2f} cm / {distance_mm:.1f} mm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_offset += line_height
        action = metrics.get('action', 'Unknown')
        confidence = metrics.get('confidence', 0)
        action_color = metrics.get('action_color', (255, 255, 255))
        cv2.putText(panel, f"ACTION: {action} ({confidence*100:.0f}%)", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)

        y_offset += line_height
        shadow_area = metrics.get('shadow_area', 0)
        cv2.putText(panel, f"Shadow Area: {shadow_area} px", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y_offset += line_height
        shadow_sharpness = metrics.get('shadow_sharpness', 0)
        cv2.putText(panel, f"Shadow Sharpness: {shadow_sharpness:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y_offset += line_height
        light_angle = metrics.get('light_angle', 0)
        light_intensity = metrics.get('light_intensity', 0)
        cv2.putText(panel, f"Light: {light_angle:.0f}deg, Intensity: {light_intensity:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y_offset += line_height + 5
        cv2.putText(panel, "Physics Breakdown:", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

        y_offset += line_height
        geo_dist = metrics.get('geometric_dist', 0)
        cv2.putText(panel, f"  Geometric: {geo_dist:.2f} cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        y_offset += line_height
        shadow_dist = metrics.get('shadow_dist', 0)
        cv2.putText(panel, f"  Shadow-based: {shadow_dist:.2f} cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        y_offset += line_height
        eucl_dist = metrics.get('euclidean_dist', 0)
        cv2.putText(panel, f"  Euclidean: {eucl_dist:.2f} cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        y_offset += line_height + 5
        threshold_text = f"Threshold: {self.touching_threshold_cm} cm (Touching) / {self.eating_threshold_cm} cm (Eating)"
        threshold_color = (0, 0, 255) if distance_cm <= self.touching_threshold_cm else (0, 255, 0)
        cv2.putText(panel, threshold_text, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, threshold_color, 1)

        fps = metrics.get('fps', 0)
        cv2.putText(panel, f"FPS: {fps:.1f}", 
                   (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.vstack([frame, panel])

        frame[:] = combined[:h, :]

def main():
    """Main application with enhanced UI"""
    print("=" * 80)
    print("SHADOW-DEPTH ACTION RECOGNITION SYSTEM - PHYSICS-BASED IMPLEMENTATION")
    print("=" * 80)
    print("\nInitializing advanced physics-based analyzer...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. No other application is using the camera")
        print("  3. Camera permissions are granted")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"\nCamera initialized: {int(actual_width)}x{int(actual_height)}")

    analyzer = PhysicsBasedShadowAnalyzer()

    print("\n" + "=" * 80)
    print("SYSTEM READY!")
    print("=" * 80)
    print("\n📋 INSTRUCTIONS:")
    print("  • Position yourself with SINGLE light source (desk lamp recommended)")
    print("  • Move your hand towards your face slowly")
    print("  • Watch real-time distance calculations in cm and mm")
    print("  • Action triggers at < 2cm (TOUCHING) or < 3cm (EATING)")
    print("\n⌨️  CONTROLS:")
    print("  [Q] - Quit application")
    print("  [S] - Save current frame and heatmap")
    print("  [R] - Reset tracking history")
    print("  [+/-] - Adjust distance threshold")
    print("\n" + "=" * 80)
    print("\n⚡ Starting real-time processing...\n")

    frame_count = 0
    saved_count = 0

    cv2.namedWindow('Shadow-Depth Action Recognition', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Intensity Matrix (Shadow Physics)', cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame from camera")
                break

            frame = cv2.flip(frame, 1)

            processed_frame, heatmap, metrics = analyzer.process_frame(frame)

            cv2.putText(processed_frame, "Physics-Based Vision System", 
                       (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow('Shadow-Depth Action Recognition', processed_frame)

            if heatmap.shape[0] > 0:
                heatmap_display = heatmap.copy()

                title_area = np.zeros((40, heatmap_display.shape[1], 3), dtype=np.uint8)
                cv2.putText(title_area, "Shadow Intensity Matrix - Physics Analysis", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                heatmap_with_title = np.vstack([title_area, heatmap_display])
                cv2.imshow('Intensity Matrix (Shadow Physics)', heatmap_with_title)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n🛑 Shutting down...")
                break

            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'capture_{timestamp}.jpg', processed_frame)
                cv2.imwrite(f'heatmap_{timestamp}.jpg', heatmap)
                saved_count += 1
                print(f"💾 Saved frame #{saved_count} - {timestamp}")

                if metrics:
                    with open(f'metrics_{timestamp}.txt', 'w') as f:
                        f.write("SHADOW-DEPTH METRICS\n")
                        f.write("=" * 40 + "\n")
                        for key, value in metrics.items():
                            f.write(f"{key}: {value}\n")
                    print(f"📊 Saved metrics - {timestamp}")

            elif key == ord('r'):
                analyzer.action_history.clear()
                analyzer.distance_history.clear()
                analyzer.shadow_area_history.clear()
                print("🔄 Reset tracking history")

            elif key == ord('+') or key == ord('='):
                analyzer.distance_threshold_cm += 0.5
                print(f"📏 Threshold increased to {analyzer.distance_threshold_cm:.1f} cm")

            elif key == ord('-') or key == ord('_'):
                analyzer.distance_threshold_cm = max(1.0, analyzer.distance_threshold_cm - 0.5)
                print(f"📏 Threshold decreased to {analyzer.distance_threshold_cm:.1f} cm")

            frame_count += 1

            if frame_count % 100 == 0:
                fps = metrics.get('fps', 0) if metrics else 0
                print(f"📊 Frame {frame_count} | FPS: {fps:.1f} | Saved: {saved_count}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:

        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")
        print(f"📊 Total frames processed: {frame_count}")
        print(f"💾 Total frames saved: {saved_count}")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()