import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

class ImageMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.metrics = {}

    def template_matching(self, image, template):
        """
        Perform template matching using cv2.matchTemplate
        """
        start_time = time.time()
        
        # Convert images to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Apply template matching
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Get the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get template dimensions
        h, w = template.shape[:2]
        
        # Define the rectangle corners
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Calculate metrics
        execution_time = time.time() - start_time
        confidence = max_val
        
        self.metrics['template'] = {
            'execution_time': execution_time,
            'confidence': confidence,
            'method': 'Template Matching'
        }
        
        return [(top_left, bottom_right)], confidence

    def feature_matching(self, image, template):
        """
        Perform feature matching using SIFT
        """
        start_time = time.time()
        
        # Convert images to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(template_gray, None)
        kp2, des2 = self.sift.detectAndCompute(img_gray, None)
        
        # Store number of keypoints
        num_keypoints_template = len(kp1)
        num_keypoints_image = len(kp2)
        
        # FLANN parameters and matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        match_ratio = len(good_matches) / len(matches) if matches else 0
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                h, w = template_gray.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                
                # Convert to list of corner points
                corners = [(tuple(map(int, corner[0]))) for corner in dst]
                
                # Calculate metrics
                execution_time = time.time() - start_time
                inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
                
                self.metrics['feature'] = {
                    'execution_time': execution_time,
                    'num_keypoints_template': num_keypoints_template,
                    'num_keypoints_image': num_keypoints_image,
                    'num_good_matches': len(good_matches),
                    'match_ratio': match_ratio,
                    'inlier_ratio': inlier_ratio,
                    'method': 'SIFT Feature Matching'
                }
                
                return [corners], match_ratio
        
        self.metrics['feature'] = {
            'execution_time': time.time() - start_time,
            'num_keypoints_template': num_keypoints_template,
            'num_keypoints_image': num_keypoints_image,
            'num_good_matches': 0,
            'match_ratio': 0,
            'inlier_ratio': 0,
            'method': 'SIFT Feature Matching'
        }
        
        return None, 0

    def draw_matches(self, image, matches, method="template"):
        """
        Draw matching results on the image
        """
        result = image.copy()
        if matches[0] is not None:
            if method == "template":
                top_left, bottom_right = matches[0][0]
                cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
            else:
                corners = matches[0][0]
                for i in range(4):
                    cv2.line(result, corners[i], corners[(i+1)%4], (0, 255, 0), 2)
                    
        return result

    def get_metrics(self):
        """
        Return the performance metrics
        """
        return self.metrics