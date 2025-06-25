# 🏥 Skin Disease Detection System

A comprehensive machine learning system for detecting and classifying 23 different types of skin diseases using deep learning and computer vision.

## 🌟 Features

- **23 Disease Categories**: Comprehensive coverage of common skin conditions
- **High Accuracy**: Powered by ResNet50V2 transfer learning
- **Easy-to-Use Interface**: Streamlit web application with drag-and-drop functionality
- **Detailed Analysis**: Confidence scores and probability distributions
- **Professional UI**: Modern, responsive design with medical-grade interface

## 📋 Supported Diseases

1. **Acne and Rosacea Photos**
2. **Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions**
3. **Atopic Dermatitis Photos**
4. **Bullous Disease Photos**
5. **Cellulitis Impetigo and other Bacterial Infections**
6. **Eczema Photos**
7. **Exanthems and Drug Eruptions**
8. **Hair Loss Photos Alopecia and other Hair Diseases**
9. **Herpes HPV and other STDs Photos**
10. **Light Diseases and Disorders of Pigmentation**
11. **Lupus and other Connective Tissue diseases**
12. **Melanoma Skin Cancer Nevi and Moles**
13. **Nail Fungus and other Nail Disease**
14. **Poison Ivy Photos and other Contact Dermatitis**
15. **Psoriasis pictures Lichen Planus and related diseases**
16. **Scabies Lyme Disease and other Infestations and Bites**
17. **Seborrheic Keratoses and other Benign Tumors**
18. **Systemic Disease**
19. **Tinea Ringworm Candidiasis and other Fungal Infections**
20. **Urticaria Hives**
21. **Vascular Tumors**
22. **Vasculitis Photos**
23. **Warts Molluscum and other Viral Infections**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset structure**:
   ```
   dataset/
   ├── train/
   │   ├── Acne and Rosacea Photos/
   │   ├── Atopic Dermatitis Photos/
   │   └── ... (23 disease categories)
   └── test/
       ├── Acne and Rosacea Photos/
       ├── Atopic Dermatitis Photos/
       └── ... (23 disease categories)
   ```

## 🎯 Usage

### 1. Training the Model

Train the model using your dataset:

```bash
python skin_disease_model.py
```

This will:
- Load and preprocess the training data
- Build a ResNet50V2-based model with transfer learning
- Train the model with data augmentation
- Evaluate on test data
- Save the trained model as `skin_disease_model.h5`
- Generate training history and confusion matrix plots

### 2. Web Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Features:
- **Home**: Overview and information about the system
- **Upload & Predict**: Upload images and get instant predictions
- **Model Information**: Technical details about the model
- **About**: Information about the application

### 3. Command Line Testing

Test individual images or directories:

```bash
# Test a single image
python test_image.py --image path/to/image.jpg

# Test all images in a directory
python test_image.py --dir path/to/image/directory

# Use a specific model file
python test_image.py --image path/to/image.jpg --model my_model.h5
```

## 🏗️ Model Architecture

### Base Model
- **ResNet50V2**: Pre-trained on ImageNet
- **Transfer Learning**: Leverages pre-trained weights for better performance

### Custom Layers
```
ResNet50V2 (base)
├── Global Average Pooling
├── Dropout (0.5)
├── Dense (512, ReLU)
├── Dropout (0.3)
├── Dense (256, ReLU)
├── Dropout (0.2)
└── Dense (23, Softmax)
```

### Training Strategy
- **Two-phase training**: Feature extraction + Fine-tuning
- **Data augmentation**: Rotation, shift, shear, zoom, flip
- **Regularization**: Dropout layers to prevent overfitting
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

## 📊 Performance

The model achieves high accuracy across multiple skin disease categories with:
- **Transfer Learning**: Leverages ResNet50V2 pre-trained weights
- **Data Augmentation**: Improves generalization
- **Regularization**: Prevents overfitting
- **Fine-tuning**: Optimizes for specific disease detection

## 🔧 Configuration

### Model Parameters
- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001 (initial), 0.0001 (fine-tuning)
- **Epochs**: 30 (initial training) + 20 (fine-tuning)

### Data Augmentation
- **Rotation**: ±20 degrees
- **Width/Height Shift**: ±20%
- **Shear**: ±20%
- **Zoom**: ±20%
- **Horizontal Flip**: Enabled

## 📁 Project Structure

```
├── dataset/
│   ├── train/          # Training images (23 categories)
│   └── test/           # Test images (23 categories)
├── skin_disease_model.py    # Main training script
├── app.py                   # Streamlit web application
├── test_image.py            # Command line testing script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── skin_disease_model.h5   # Trained model (generated)
├── training_history.png    # Training plots (generated)
└── confusion_matrix.png    # Confusion matrix (generated)
```

## ⚠️ Important Notes

### Medical Disclaimer
- **This tool is for educational purposes only**
- **Not a replacement for professional medical advice**
- **Always consult healthcare professionals for diagnosis**
- **Results should be validated by medical experts**

### Limitations
- Accuracy depends on image quality and lighting
- Some rare conditions may not be accurately classified
- Model performance varies across different skin types
- Results are probabilistic and not definitive

### Best Practices
- Use clear, well-lit images
- Ensure good image resolution
- Include the affected area clearly
- Avoid shadows and reflections
- Use consistent lighting conditions

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   # Install CPU-only TensorFlow
   pip install tensorflow-cpu
   ```

2. **Memory Issues**:
   - Reduce batch size in `skin_disease_model.py`
   - Use smaller image size
   - Close other applications

3. **Model Not Found**:
   - Ensure you've trained the model first
   - Check file path in `app.py`

4. **Dataset Issues**:
   - Verify folder structure matches expected format
   - Ensure all categories have images
   - Check file permissions

## 🔮 Future Enhancements

- [ ] Mobile application development
- [ ] Real-time video analysis
- [ ] Integration with electronic health records
- [ ] Additional disease categories
- [ ] Multi-language support
- [ ] API development for third-party integration

## 📞 Support

For questions, issues, or contributions:
- Contact me at Github: https://github.com/maheshvellogi99 or at Linkedin: http://linkedin.com/in/mahesh-vellogi-aa13522ba
- Check the troubleshooting section
- Review the code comments
- Ensure all dependencies are installed
- Verify dataset structure

## 📄 License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding medical software and data privacy.

---

**Note**: This system is designed to assist healthcare professionals and should not be used as a standalone diagnostic tool. Always consult qualified medical professionals for proper diagnosis and treatment. 