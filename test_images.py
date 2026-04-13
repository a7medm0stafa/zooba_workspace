import cv2
import yaml
import sys
import os

sys.path.insert(0, 'src/perception/perception/nodes')
# Just importing won't work easily if we rely on ROS2 parameters. Let's just copy the class.
from sign_detection_node import SignDetector

with open('src/perception/config/sign_detection_params.yaml', 'r') as f:
    config = yaml.safe_load(f)
p = config['perception']['sign_detection_node']['ros__parameters']

detector = SignDetector(p)

for img_name in ['left.jpg', 'right.jpg', 'slow.PNG', 'stop.PNG']:
    path = img_name
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        continue
    
    cmd, conf, bbox, img_bgr, clean_masks, dbg_shapes, dbg_edges, timings = detector.detect(img, show_dbg=True)
    print(f"[{img_name}] Output = {cmd} (conf: {conf}) | BBOX: {bbox}")
    
    # Save debug shapes
    if dbg_shapes is not None:
        cv2.imwrite(f"{img_name}_shapes.jpg", dbg_shapes)
    if clean_masks is not None:
        r, y, b = clean_masks
        cv2.imwrite(f"{img_name}_mask_r.jpg", r)
        cv2.imwrite(f"{img_name}_mask_y.jpg", y)
        cv2.imwrite(f"{img_name}_mask_b.jpg", b)
