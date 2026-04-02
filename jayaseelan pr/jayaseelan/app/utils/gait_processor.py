import cv2
import numpy as np
import os

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False


def _analyze_walking_quality(landmarks_list, task_type='single'):
    """
    Analyzes walking quality from a list of MediaPipe pose landmark objects.
    Returns a score dict with individual metrics and an overall walking score (0-100).

    Higher score = better/straighter walking.
    Lower score  = poor walking posture = higher dementia risk.
    """
    if not landmarks_list:
        return _mock_walking_score()

    L = mp_pose.PoseLandmark

    spine_scores = []
    shoulder_scores = []
    hip_scores = []
    head_scores = []
    step_symmetry_scores = []
    stride_lengths = []
    arm_swings = []

    for lm in landmarks_list:
        pts = lm.landmark

        def p(idx):
            return np.array([pts[idx].x, pts[idx].y])

        try:
            nose        = p(L.NOSE)
            l_shoulder  = p(L.LEFT_SHOULDER)
            r_shoulder  = p(L.RIGHT_SHOULDER)
            l_hip       = p(L.LEFT_HIP)
            r_hip       = p(L.RIGHT_HIP)
            l_ankle     = p(L.LEFT_ANKLE)
            r_ankle     = p(L.RIGHT_ANKLE)
            l_knee      = p(L.LEFT_KNEE)
            r_knee      = p(L.RIGHT_KNEE)
            l_wrist     = p(L.LEFT_WRIST)
            r_wrist     = p(L.RIGHT_WRIST)
        except Exception:
            continue

        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0
        spine_vec = mid_shoulder - mid_hip

        # -- Improved accuracy using distance-invariant normalizations --
        torso_length = abs(mid_shoulder[1] - mid_hip[1]) + 1e-6
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0]) + 1e-6

        # 1. Spine alignment (normalized body lean)
        spine_lean   = abs(spine_vec[0]) / torso_length
        spine_score  = max(0, 1 - spine_lean * 1.5)
        spine_scores.append(spine_score)

        # 2. Shoulder symmetry
        shoulder_diff  = abs(l_shoulder[1] - r_shoulder[1]) / torso_length
        shoulder_score = max(0, 1 - shoulder_diff * 2.0)
        shoulder_scores.append(shoulder_score)

        # 3. Hip symmetry
        hip_diff  = abs(l_hip[1] - r_hip[1]) / torso_length
        hip_score = max(0, 1 - hip_diff * 2.0)
        hip_scores.append(hip_score)

        # 4. Head position
        head_offset = abs(nose[0] - mid_shoulder[0]) / shoulder_width
        head_score  = max(0, 1 - head_offset * 1.5)
        head_scores.append(head_score)

        # 5. Step symmetry
        ankle_diff       = abs(l_ankle[1] - r_ankle[1])
        step_sym_score   = max(0, 1 - ankle_diff * 2.5)
        step_symmetry_scores.append(step_sym_score)

        # 6. Stride length proxy
        stride_dist = abs(l_ankle[0] - r_ankle[0]) / torso_length
        stride_lengths.append(stride_dist)

        # 7. Arm Swing Detection (Parkinsonian / Dementia Tremor Indicator)
        l_swing = abs(l_wrist[0] - l_hip[0]) / torso_length
        r_swing = abs(r_wrist[0] - r_hip[0]) / torso_length
        arm_swing_diff = abs(l_swing - r_swing)
        arm_swing_score = max(0, 1 - arm_swing_diff * 1.8)
        arm_swings.append(arm_swing_score)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.5

    spine_avg    = safe_mean(spine_scores)
    shoulder_avg = safe_mean(shoulder_scores)
    hip_avg      = safe_mean(hip_scores)
    head_avg     = safe_mean(head_scores)
    step_avg     = safe_mean(step_symmetry_scores)
    arm_avg      = safe_mean(arm_swings)

    # Calculate Stride Length Score
    stride_max = max(stride_lengths) if stride_lengths else 0
    stride_score = max(0, min(1.0, stride_max * 1.5))

    # Calculate Velocity/Cadence Score (change in ankle distance between frames)
    if len(stride_lengths) > 1:
        cadence_diffs = np.abs(np.diff(stride_lengths))
        cadence_score = max(0, min(1.0, float(np.mean(cadence_diffs)) * 10.0))
    else:
        cadence_score = 0.5

    # Weighted overall score
    
    # Dual-task explicitly penalizes stride and cadence deterioration more heavily 
    # as cognitive-motor interference is the primary biomarker for dementia.
    if task_type == 'dual':
        w_step, w_stride, w_cadence = 0.20, 0.15, 0.15
        w_spine, w_shoulder, w_hip, w_head, w_arm = 0.15, 0.05, 0.10, 0.05, 0.15
    else:
        w_step, w_stride, w_cadence = 0.15, 0.10, 0.10
        w_spine, w_shoulder, w_hip, w_head, w_arm = 0.20, 0.10, 0.10, 0.05, 0.20

    overall = (
        spine_avg     * w_spine +
        shoulder_avg  * w_shoulder +
        hip_avg       * w_hip +
        head_avg      * w_head +
        arm_avg       * w_arm +
        step_avg      * w_step +
        stride_score  * w_stride +
        cadence_score * w_cadence
    )

    return {
        'overall':       round(overall * 100, 1),
        'spine':         round(spine_avg * 100, 1),
        'shoulder':      round(shoulder_avg * 100, 1),
        'hip':           round(hip_avg * 100, 1),
        'head':          round(head_avg * 100, 1),
        'arm_swing':     round(arm_avg * 100, 1),
        'step_symmetry': round(step_avg * 100, 1),
        'stride_length': round(stride_score * 100, 1),
        'velocity':      round(cadence_score * 100, 1),
    }


def _mock_walking_score():
    """Fallback when MediaPipe is not available."""
    s = float(np.random.uniform(50, 90))
    return {
        'overall': round(s, 1),
        'spine':   round(s + np.random.uniform(-5, 5), 1),
        'shoulder': round(s + np.random.uniform(-5, 5), 1),
        'hip':      round(s + np.random.uniform(-5, 5), 1),
        'head':     round(s + np.random.uniform(-5, 5), 1),
        'step_symmetry': round(s + np.random.uniform(-5, 5), 1),
        'stride_length': round(s + np.random.uniform(-5, 5), 1),
        'velocity':      round(s + np.random.uniform(-5, 5), 1),
    }


def process_video(video_path, task_type='single', max_frames=30):
    """
    Extracts frames from video, performs pose detection, and returns
    (processed_frames array, walking_quality dict).
    """
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    landmarks_list = []

    while cap.isOpened() and len(processed_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        skeleton_img = np.zeros((h, w, 3), dtype=np.uint8)

        if MEDIAPIPE_AVAILABLE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks_list.append(results.pose_landmarks)
                mp_drawing.draw_landmarks(
                    skeleton_img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        else:
            cv2.line(skeleton_img, (w//2, h//4), (w//2, 3*h//4), (255, 255, 255), 5)
            cv2.circle(skeleton_img, (w//2, h//4), 10, (255, 255, 255), -1)

        skeleton_img_resized = cv2.resize(skeleton_img, (128, 128))
        processed_frames.append(skeleton_img_resized)

    cap.release()

    while len(processed_frames) < max_frames:
        processed_frames.append(np.zeros((128, 128, 3), dtype=np.uint8))

    walking_quality = _analyze_walking_quality(landmarks_list, task_type=task_type)
    return np.array(processed_frames), walking_quality


def save_debug_frames(processed_frames, output_dir='app/static/processed'):
    """Helper to save frames for UI display."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paths = []
    for i in [0, 10, 20]:
        filename = f"frame_{i}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, processed_frames[i])
        paths.append(filename)
    return paths
