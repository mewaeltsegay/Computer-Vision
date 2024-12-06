import requests
import os
from pathlib import Path
import time
import cv2
import numpy as np
import random

UNSPLASH_ACCESS_KEY = "0Am44T3KkyFEjQoB4Fc57xvsx8uurnPqWP3u8DG4EFY"
UNSPLASH_API_URL = "https://api.unsplash.com/photos/random"

def apply_transformation(image, transform_type):
    """
    Apply random transformations to the image based on transform_type
    """
    if transform_type == "rotation":
        # Random rotation between 15 and 45 degrees
        angle = random.uniform(15, 45)
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
        
    elif transform_type == "scale":
        # Random scale between 0.5 and 1.5
        scale = random.uniform(0.5, 1.5)
        return cv2.resize(image, None, fx=scale, fy=scale)
        
    elif transform_type == "lighting":
        # Adjust brightness and contrast
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
    elif transform_type == "perspective":
        # Random perspective transform
        height, width = image.shape[:2]
        src_pts = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
        offset = width * 0.2
        dst_pts = np.float32([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [width-1-random.uniform(0, offset), random.uniform(0, offset)],
            [random.uniform(0, offset), height-1-random.uniform(0, offset)],
            [width-1-random.uniform(0, offset), height-1-random.uniform(0, offset)]
        ])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, matrix, (width, height))
        
    elif transform_type == "occlusion":
        # Add random rectangle occlusion
        height, width = image.shape[:2]
        x = random.randint(0, width//2)
        y = random.randint(0, height//2)
        w = random.randint(width//4, width//2)
        h = random.randint(height//4, height//2)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), -1)
        return image
        
    elif transform_type == "noise":
        # Add Gaussian noise
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
        
    return image

def interactive_crop_with_transform(image_path, template_path, transform_type=None):
    """
    Allow user to select region and apply transformation
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False
        
    window_name = "Select template region (drag mouse to select, press ENTER when done)"
    cv2.namedWindow(window_name)
    rect = cv2.selectROI(window_name, image, False)
    cv2.destroyWindow(window_name)
    
    if rect[2] > 0 and rect[3] > 0:
        # Crop the template
        template = image[int(rect[1]):int(rect[1]+rect[3]), 
                       int(rect[0]):int(rect[0]+rect[2])]
        
        # Apply transformation if specified
        if transform_type:
            template = apply_transformation(template, transform_type)
            
        cv2.imwrite(template_path, template)
        print(f"Created transformed template {template_path}")
        return True
    
    return False

def download_image(query, filename, min_size=(800, 600)):
    """
    Download image from Unsplash API
    """
    params = {
        'query': query,
        'client_id': UNSPLASH_ACCESS_KEY,
        'orientation': 'landscape'
    }
    
    try:
        response = requests.get(UNSPLASH_API_URL, params=params)
        response.raise_for_status()
        
        image_data = response.json()
        image_url = image_data['urls']['regular']
        
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(image_response.content)
            
        # Verify the image was downloaded correctly and meets minimum size
        img = cv2.imread(filename)
        if img is None:
            print(f"Failed to verify downloaded image {filename}")
            return False
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            print(f"Image too small: {img.shape[:2]}, minimum required: {min_size}")
            return False
            
        print(f"Successfully downloaded {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return False

def setup_test_images():
    Path('images').mkdir(exist_ok=True)
    
    test_scenes = [
        # Basic tests
        ('organized desk workspace', 'basic_scene.jpg', None),
        ('keyboard on desk', 'rotation_scene.jpg', 'rotation'),
        
        # Scale variations
        ('street with traffic signs', 'scale_scene_far.jpg', 'scale_down'),
        ('street with traffic signs', 'scale_scene_close.jpg', 'scale_up'),
        
        # Lighting variations
        ('room with lamp natural light', 'lighting_scene_bright.jpg', 'lighting_bright'),
        ('room with lamp natural light', 'lighting_scene_dark.jpg', 'lighting_dark'),
        
        # Occlusion tests
        ('coffee shop counter cups', 'occlusion_scene_partial.jpg', 'occlusion_small'),
        ('coffee shop counter cups', 'occlusion_scene_major.jpg', 'occlusion_large'),
        
        # Multiple similar objects
        ('classroom with chairs', 'multiple_scene.jpg', None),
        
        # Perspective changes
        ('modern building windows', 'perspective_scene_slight.jpg', 'perspective_mild'),
        ('modern building windows', 'perspective_scene_extreme.jpg', 'perspective_extreme'),
        
        # Texture variations
        ('detailed fabric pattern', 'texture_scene_regular.jpg', None),
        ('detailed fabric pattern', 'texture_scene_complex.jpg', 'noise'),
        
        # Combined challenges
        ('messy desk workspace', 'complex_scene_moderate.jpg', ['rotation', 'lighting']),
        ('messy desk workspace', 'complex_scene_extreme.jpg', ['rotation', 'scale', 'lighting'])
    ]
    
    successful_downloads = []
    
    for query, scene_name, transform_type in test_scenes:
        scene_path = os.path.join('images', scene_name)
        template_name = scene_name.replace('scene', 'template')
        template_path = os.path.join('images', template_name)
        
        print(f"\nProcessing test case: {scene_name}")
        print(f"This image will test: {query}")
        if transform_type:
            print(f"Will apply {transform_type} transformation to template")
        
        if download_image(query, scene_path):
            print("\nPlease select the region for the template:")
            if interactive_crop_with_transform(scene_path, template_path, transform_type):
                successful_downloads.append((scene_name, template_name))
            time.sleep(1)
    
    print("\nSuccessfully created the following test pairs:")
    for scene, template in successful_downloads:
        print(f"Scene: {scene} -> Template: {template}")
    
    print("\nUpdate your test_matcher.py test cases with:")
    print("test_cases = [")
    for scene, template in successful_downloads:
        print(f'    ("images/{scene}", "images/{template}"),')
    print("]")

if __name__ == "__main__":
    print("Starting to download test images...")
    setup_test_images()
    print("\nDownload process completed!") 