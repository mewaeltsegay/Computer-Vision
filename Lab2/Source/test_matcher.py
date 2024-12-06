import cv2
from image_matcher import ImageMatcher
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def visualize_results(image_path, template_path, template_result, feature_result):
    """
    Display original images and results side by side
    """
    # Read images
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(cv2.imread(str(template_path)), cv2.COLOR_BGR2RGB)
    template_result = cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB)
    feature_result = cv2.cvtColor(feature_result, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot images
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(template)
    axes[0, 1].set_title('Template')
    axes[1, 0].imshow(template_result)
    axes[1, 0].set_title('Template Matching Result')
    axes[1, 1].imshow(feature_result)
    axes[1, 1].set_title('Feature Matching Result')
    
    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def analyze_results(results):
    """
    Analyze and compare the performance of both methods
    """
    template_scores = []
    feature_scores = []
    template_times = []
    feature_times = []
    
    print("\nDetailed Analysis:")
    print("=" * 80)
    
    for result in results:
        template_scores.append(result['template_score'])
        feature_scores.append(result['feature_score'])
        template_times.append(result['metrics']['template']['execution_time'])
        feature_times.append(result['metrics']['feature']['execution_time'])
        
        print(f"\nTest Case: {result['image']}")
        print("-" * 40)
        
        # Template Matching Results
        print("Template Matching:")
        print(f"  Confidence: {result['template_score']:.3f}")
        print(f"  Execution Time: {result['metrics']['template']['execution_time']:.3f}s")
        
        # Feature Matching Results
        print("\nFeature Matching:")
        print(f"  Match Ratio: {result['feature_score']:.3f}")
        print(f"  Keypoints (Template/Image): {result['metrics']['feature']['num_keypoints_template']}/{result['metrics']['feature']['num_keypoints_image']}")
        print(f"  Good Matches: {result['metrics']['feature']['num_good_matches']}")
        print(f"  Inlier Ratio: {result['metrics']['feature']['inlier_ratio']:.3f}")
        print(f"  Execution Time: {result['metrics']['feature']['execution_time']:.3f}s")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print("\nTemplate Matching:")
    print(f"  Average Confidence: {np.mean(template_scores):.3f} ± {np.std(template_scores):.3f}")
    print(f"  Average Time: {np.mean(template_times):.3f}s ± {np.std(template_times):.3f}s")
    
    print("\nFeature Matching:")
    print(f"  Average Match Ratio: {np.mean(feature_scores):.3f} ± {np.std(feature_scores):.3f}")
    print(f"  Average Time: {np.mean(feature_times):.3f}s ± {np.std(feature_times):.3f}s")

def test_matching(image_path, template_path, output_dir):
    """
    Test matching algorithms on a pair of images
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get base name first
    base_name = Path(image_path).stem
    
    # Read images with error checking
    image = cv2.imread(str(image_path))
    template = cv2.imread(str(template_path))
    
    if image is None or template is None:
        print(f"Error: Could not read images:\n  {image_path}\n  {template_path}")
        return None, None, None
    
    # Initialize matcher
    matcher = ImageMatcher()
    
    try:
        # Test template matching
        template_matches, template_score = matcher.template_matching(image, template)
        template_result = matcher.draw_matches(image, (template_matches, template_score), "template")
        
        # Test feature matching
        feature_matches, feature_score = matcher.feature_matching(image, template)
        feature_result = matcher.draw_matches(image, (feature_matches, feature_score), "feature")
        
        # Get metrics
        metrics = matcher.get_metrics()
        
        # Save individual results
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_template.jpg"), template_result)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_feature.jpg"), feature_result)
        
        # Add visualization and save comparison
        fig = visualize_results(image_path, template_path, template_result, feature_result)
        fig.savefig(str(Path(output_dir) / f"{base_name}_comparison.jpg"))
        plt.close(fig)
        
        return template_score, feature_score, metrics
        
    except Exception as e:
        print(f"Error processing images {image_path} and {template_path}: {str(e)}")
        return None, None, None

def main():
    # Define test cases with different scenarios
    test_cases = [
        # 1. Basic template matching
        ("images/basic_scene.jpg", "images/basic_template.jpg"),
        
        # 2. Rotation test
        ("images/rotation_scene.jpg", "images/rotation_template.jpg"),
        
        # 3. Scale variations
        ("images/scale_scene_far.jpg", "images/scale_template_far.jpg"),
        ("images/scale_scene_close.jpg", "images/scale_template_close.jpg"),
        
        # 4. Lighting variations
        ("images/lighting_scene_bright.jpg", "images/lighting_template_bright.jpg"),
        ("images/lighting_scene_dark.jpg", "images/lighting_template_dark.jpg"),
        
        # 5. Occlusion tests
        ("images/occlusion_scene_partial.jpg", "images/occlusion_template_partial.jpg"),
        ("images/occlusion_scene_major.jpg", "images/occlusion_template_major.jpg"),
        
        # 6. Multiple similar objects
        ("images/multiple_scene.jpg", "images/multiple_template.jpg"),
        
        # 7. Perspective changes
        ("images/perspective_scene_slight.jpg", "images/perspective_template_slight.jpg"),
        ("images/perspective_scene_extreme.jpg", "images/perspective_template_extreme.jpg"),
        
        # 8. Texture variations
        ("images/texture_scene_regular.jpg", "images/texture_template_regular.jpg"),
        ("images/texture_scene_complex.jpg", "images/texture_template_complex.jpg"),
        
        # 9. Combined challenges
        ("images/complex_scene_moderate.jpg", "images/complex_template_moderate.jpg"),
        ("images/complex_scene_extreme.jpg", "images/complex_template_extreme.jpg")
    ]
    
    output_dir = "results"
    
    results = []
    for image_path, template_path in test_cases:
        # Check if both files exist
        if not Path(image_path).exists() or not Path(template_path).exists():
            print(f"Skipping missing files:\n  {image_path}\n  {template_path}")
            continue
            
        print(f"\nProcessing: {Path(image_path).name}")
        template_score, feature_score, metrics = test_matching(image_path, template_path, output_dir)
        
        if template_score is not None:
            results.append({
                'image': image_path,
                'template': template_path,
                'template_score': template_score,
                'feature_score': feature_score,
                'metrics': metrics
            })
    
    if results:
        # Analyze results
        analyze_results(results)
    else:
        print("\nNo valid test cases were processed.")
        print("Please run download_test_images.py first to generate the test images.")

if __name__ == "__main__":
    main() 