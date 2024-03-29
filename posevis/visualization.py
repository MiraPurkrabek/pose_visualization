import numpy as np
import cv2
from copy import deepcopy

NEUTRAL_COLOR = (52, 235, 107)

LEFT_ARM_COLOR = (216, 235, 52)
LEFT_LEG_COLOR = (235, 107, 52)
LEFT_SIDE_COLOR = (245, 188, 113)
LEFT_FACE_COLOR = (235, 52, 107)

RIGHT_ARM_COLOR = (52, 235, 216)
RIGHT_LEG_COLOR = (52, 107, 235)
RIGHT_SIDE_COLOR = (52, 171, 235)
RIGHT_FACE_COLOR = (107, 52, 235)

COCO_MARKERS = [
    ["nose", cv2.MARKER_CROSS, NEUTRAL_COLOR],
    ["left_eye", cv2.MARKER_SQUARE, LEFT_FACE_COLOR],
    ["right_eye", cv2.MARKER_SQUARE, RIGHT_FACE_COLOR],
    ["left_ear", cv2.MARKER_CROSS, LEFT_FACE_COLOR],
    ["right_ear", cv2.MARKER_CROSS, RIGHT_FACE_COLOR],
    ["left_shoulder", cv2.MARKER_TRIANGLE_UP, LEFT_ARM_COLOR],
    ["right_shoulder", cv2.MARKER_TRIANGLE_UP, RIGHT_ARM_COLOR],
    ["left_elbow", cv2.MARKER_SQUARE, LEFT_ARM_COLOR],
    ["right_elbow", cv2.MARKER_SQUARE, RIGHT_ARM_COLOR],
    ["left_wrist", cv2.MARKER_CROSS, LEFT_ARM_COLOR],
    ["right_wrist", cv2.MARKER_CROSS, RIGHT_ARM_COLOR],
    ["left_hip", cv2.MARKER_TRIANGLE_UP, LEFT_LEG_COLOR],
    ["right_hip", cv2.MARKER_TRIANGLE_UP, RIGHT_LEG_COLOR],
    ["left_knee", cv2.MARKER_SQUARE, LEFT_LEG_COLOR],
    ["right_knee", cv2.MARKER_SQUARE, RIGHT_LEG_COLOR],
    ["left_ankle", cv2.MARKER_TILTED_CROSS, LEFT_LEG_COLOR],
    ["right_ankle", cv2.MARKER_TILTED_CROSS, RIGHT_LEG_COLOR],
]


COCO_SKELETON = [
    [[16, 14], LEFT_LEG_COLOR],   # Left ankle - Left knee
    [[14, 12], LEFT_LEG_COLOR],   # Left knee - Left hip
    [[17, 15], RIGHT_LEG_COLOR],   # Right ankle - Right knee
    [[15, 13], RIGHT_LEG_COLOR],   # Right knee - Right hip
    [[12, 13], NEUTRAL_COLOR],   # Left hip - Right hip
    [[ 6, 12], LEFT_SIDE_COLOR],   # Left hip - Left shoulder
    [[ 7, 13], RIGHT_SIDE_COLOR],   # Right hip - Right shoulder
    [[ 6,  7], NEUTRAL_COLOR],   # Left shoulder - Right shoulder
    [[ 6,  8], LEFT_ARM_COLOR],   # Left shoulder - Left elbow
    [[ 7,  9], RIGHT_ARM_COLOR],   # Right shoulder - Right elbow
    [[ 8, 10], LEFT_ARM_COLOR],   # Left elbow - Left wrist
    [[ 9, 11], RIGHT_ARM_COLOR],   # Right elbow - Right wrist
    [[ 2,  3], NEUTRAL_COLOR],   # Left eye - Right eye
    [[ 1,  2], LEFT_FACE_COLOR],   # Nose - Left eye
    [[ 1,  3], RIGHT_FACE_COLOR],   # Nose - Right eye
    [[ 2,  4], LEFT_FACE_COLOR],   # Left eye - Left ear
    [[ 3,  5], RIGHT_FACE_COLOR],   # Right eye - Right ear
    [[ 4,  6], LEFT_FACE_COLOR],   # Left ear - Left shoulder
    [[ 5,  7], RIGHT_FACE_COLOR],   # Right ear - Right shoulder
]


def draw_line(img, start, stop, color, line_type, thickness=1):
    start = np.array(start)[:2]
    stop = np.array(stop)[:2]
    if line_type.lower() == "solid":
        img = cv2.line(
            img,
            (int(start[0]), int(start[1])),
            (int(stop[0]), int(stop[1])),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif line_type.lower() == "dashed":
        delta = stop - start
        length = np.linalg.norm(delta)
        frac = np.linspace(0, 1, num=int(length/5), endpoint=True)
        for i in range(0, len(frac)-1, 2):
            s = start + frac[i] * delta
            e = start + frac[i+1] * delta
            img = cv2.line(
                img,
                (int(s[0]), int(s[1])),
                (int(e[0]), int(e[1])),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    elif line_type.lower() == "doted":
        delta = stop - start
        length = np.linalg.norm(delta)
        frac = np.linspace(0, 1, num=int(length/5), endpoint=True)
        for i in range(0, len(frac)):
            s = start + frac[i] * delta
            img = cv2.circle(
                img,
                (int(s[0]), int(s[1])),
                radius=max(thickness//2, 1),
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return img


def pose_visualization(
        img,
        keypoints,
        format="COCO",
        greyness=1.0,
        show_markers=True,
        show_bones=True,
        line_type="solid",
        width_multiplier=1.0,
        bbox_width_multiplier=1.0,
        show_bbox=False,
        differ_individuals=False,
    ):
    """
    This function draws keypoints on the image
    """
    assert line_type.lower() in ["solid", "dashed", "doted"], "line_type should be either solid, dashed or doted"
    
    if format.upper() not in ["COCO", "OCHUMAN", "CROWDPOSE"]:
        raise NotImplementedError("Only COCO, OCHuman and CrowdPose format is supported for now")

    bbox = None
    if isinstance(keypoints, dict):
        try:
            bbox = np.array(keypoints["bbox"]).flatten()
        except KeyError:
            pass
        keypoints = np.array(keypoints["keypoints"])

    if format.upper() == "OCHUMAN":
        keypoints = keypoints.reshape(-1, 3)
        keypoints = keypoints[[16, 18, 17, 15, 14, 3, 0, 4, 1, 5, 2, 9, 6, 10, 7, 11, 8], :]

        # Map visibilty to COCO format
        new_keypoints = deepcopy(keypoints)
        new_keypoints[keypoints[:, -1] == 0, -1] = 0
        new_keypoints[keypoints[:, -1] == 1, -1] = 2
        new_keypoints[keypoints[:, -1] == 2, -1] = 1
        new_keypoints[keypoints[:, -1] == 3, -1] = 1

        keypoints = new_keypoints.flatten()

    elif format.upper() == "CROWDPOSE":
        keypoints = keypoints.reshape(-1, 3)
        keypoints = keypoints[[12, 12, 12, 13, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :]
        keypoints = keypoints.flatten()

        if bbox is not None:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

    # If keypoints is a list of poses, draw them all
    if len(keypoints) % 17 != 0:
        for keypoint in keypoints:
            img = pose_visualization(
                img,
                keypoint,
                format=format,
                greyness=greyness,
                show_markers=show_markers,
                show_bones=show_bones,
                line_type=line_type,
                width_multiplier=width_multiplier,
                show_bbox=show_bbox,
                differ_individuals=differ_individuals,
            )
        return img
    
    keypoints = np.array(keypoints).reshape(17, -1)
    # If keypoint visibility is not provided, assume all keypoints are visible
    if keypoints.shape[1] == 2:
        keypoints = np.hstack([keypoints, np.ones((17, 1))*2])
    
    assert keypoints.shape[1] == 3, "Keypoints should be in the format (x, y, visibility)"
    assert keypoints.shape[0] == 17, "Keypoints should be in the format (x, y, visibility)"

    if isinstance(img, str):
        img = cv2.imread(img)

    valid_kpts = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
    num_valid_kpts = np.sum(valid_kpts)

    if num_valid_kpts == 0:
        return img

    if bbox is None:
        min_x = np.min(keypoints[keypoints[:, 0] > 0, 0])
        min_y = np.min(keypoints[keypoints[:, 1] > 0, 1])
        max_x = np.max(keypoints[keypoints[:, 0] > 0, 0])
        max_y = np.max(keypoints[keypoints[:, 1] > 0, 1])
    else:
        min_x = bbox[0]
        min_y = bbox[1]
        max_x = bbox[2]
        max_y = bbox[3]

    max_area = (max_x-min_x) * (max_y-min_y)
    line_width = max(int(np.sqrt(max_area) / 500 * width_multiplier), 1)
    marker_size = max(int(np.sqrt(max_area) / 80), 1)
    invisible_marker_size = max(int(np.sqrt(max_area) / 100), 1)
    marker_thickness = max(int(np.sqrt(max_area) / 100), 1)

    if differ_individuals:
        instance_color = np.random.randint(0, 255, size=(3, )).tolist()
        instance_color = tuple(instance_color)

    if show_bbox and not (bbox is None):
        pts = [
            (bbox[0], bbox[1]),
            (bbox[0], bbox[3]),
            (bbox[2], bbox[3]),
            (bbox[2], bbox[1]),
            (bbox[0], bbox[1]),
        ]
        for i in range(len(pts)-1):
            if differ_individuals:
                img = draw_line(img, pts[i], pts[i+1], instance_color, "doted", thickness=line_width)
            else:
                img = draw_line(img, pts[i], pts[i+1], (0, 255, 0), line_type, thickness=line_width)

    if show_markers:
        for kpt, marker_info in zip(keypoints, COCO_MARKERS):
            if kpt[0] == 0 and kpt[1] == 0:
                continue

            if differ_individuals:
                color = instance_color if kpt[2] == 2 else (140, 140, 140)
            else:
                color = marker_info[2] if kpt[2] == 2 else (140, 140, 140)

            if kpt[2] == 1:
                img_overlay = img.copy()
                img_overlay = cv2.drawMarker(
                    img_overlay,
                    (int(kpt[0]), int(kpt[1])),
                    color=color,
                    markerType=marker_info[1],
                    markerSize=marker_size,
                    thickness=marker_thickness,
                )
                img = cv2.addWeighted(img_overlay, 0.4, img, 0.6, 0)

            else:
                img = cv2.drawMarker(
                    img,
                    (int(kpt[0]), int(kpt[1])),
                    color=color,
                    markerType=marker_info[1],
                    markerSize=invisible_marker_size if kpt[2]==1 else marker_size,
                    thickness=marker_thickness,
                )

    if show_bones:
        for bone_info in COCO_SKELETON:
            kp1 = keypoints[bone_info[0][0]-1, :]
            kp2 = keypoints[bone_info[0][1]-1, :]
            
            if (kp1[0] == 0 and kp1[1] == 0) or (kp2[0] == 0 and kp2[1] == 0):
                continue
            
            dashed = kp1[2] == 1 or kp2[2] == 1
            
            if differ_individuals:
                color = np.array(instance_color)
            else:
                color = np.array(bone_info[1])
            color = (color * greyness).astype(int).tolist()

            if dashed:
                img_overlay = img.copy()
                img_overlay = draw_line(img_overlay, kp1, kp2, color, line_type, thickness=line_width)
                img = cv2.addWeighted(img_overlay, 0.4, img, 0.6, 0)

            else:
                img = draw_line(img, kp1, kp2, color, line_type, thickness=line_width)

    return img


if __name__ == "__main__":
    kpts = np.array(
        [
        344,
        222,
        2,
        356,
        211,
        2,
        330,
        211,
        2,
        372,
        220,
        2,
        309,
        224,
        2,
        413,
        279,
        2,
        274,
        300,
        2,
        444,
        372,
        2,
        261,
        396,
        2,
        398,
        359,
        2,
        316,
        372,
        2,
        407,
        489,
        2,
        185,
        580,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    )

    kpts = kpts.reshape(-1, 3)
    kpts[:, -1] = np.random.randint(1, 3, size=(17, ))

    img = pose_visualization("test.jpg", kpts, show_markers=True, line_type="solid")

    kpts2 = kpts.copy()
    kpts2[kpts2[:, 1] > 0, :2] += 10
    img = pose_visualization(img, kpts2, show_markers=False, line_type="doted")
    
    cv2.namedWindow('test', cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("test", img)
    cv2.waitKey(0)