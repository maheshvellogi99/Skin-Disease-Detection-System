import os
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
from skin_disease_model import SkinDiseaseDetector

def test_single_image(image_path, model_path='skin_disease_model.h5'):
    """
    Test a single image with the trained model
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using skin_disease_model.py")
        return
    
    # Initialize detector
    detector = SkinDiseaseDetector()
    detector.get_class_names()
    detector.load_model(model_path)
    
    # Make prediction
    try:
        predicted_class, confidence, all_predictions = detector.predict_single_image(image_path)
        
        print(f"\n=== Skin Disease Detection Results ===")
        print(f"Image: {image_path}")
        print(f"Predicted Disease: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show top 5 predictions
        print(f"\nTop 5 Predictions:")
        top_5_indices = np.argsort(all_predictions)[-5:][::-1]
        for i, idx in enumerate(top_5_indices):
            prob = all_predictions[idx]
            disease = detector.class_names[idx]
            print(f"{i+1}. {disease}: {prob:.2%}")
        
        # Confidence level
        if confidence >= 0.8:
            print(f"\n✅ High confidence prediction")
        elif confidence >= 0.6:
            print(f"\n⚠️ Medium confidence prediction")
        else:
            print(f"\n❌ Low confidence prediction")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

def test_multiple_images(image_dir, model_path='skin_disease_model.h5'):
    """
    Test multiple images from a directory
    
    Args:
        image_dir (str): Directory containing images
        model_path (str): Path to the trained model
    """
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using skin_disease_model.py")
        return
    
    # Initialize detector
    detector = SkinDiseaseDetector()
    detector.get_class_names()
    detector.load_model(model_path)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No image files found in '{image_dir}'")
        return
    
    print(f"Found {len(image_files)} images to test")
    print("=" * 50)
    
    # Test each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{i}. Testing: {os.path.basename(image_path)}")
        try:
            predicted_class, confidence, all_predictions = detector.predict_single_image(image_path)
            print(f"   Predicted: {predicted_class} (Confidence: {confidence:.2%})")
        except Exception as e:
            print(f"   Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test skin disease detection model')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    parser.add_argument('--model', type=str, default='skin_disease_model.h5', 
                       help='Path to trained model file')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image, args.model)
    elif args.dir:
        test_multiple_images(args.dir, args.model)
    else:
        print("Please provide either --image or --dir argument")
        print("Example usage:")
        print("  python test_image.py --image path/to/image.jpg")
        print("  python test_image.py --dir path/to/image/directory")

if __name__ == "__main__":
    main() 