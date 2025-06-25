import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
from skin_disease_model import SkinDiseaseDetector
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color:#555879;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        detector = SkinDiseaseDetector()
        detector.get_class_names()
        detector.load_model('skin_disease_model.h5')
        return detector
    except:
        st.error("Model not found. Please train the model first using skin_disease_model.py")
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Skin Disease Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Upload & Predict", "Model Information", "About"]
    )
    
    if page == "Home":
        st.markdown("""
        ## Welcome to the Skin Disease Detection System
        
        This application uses advanced deep learning to detect and classify various skin diseases from uploaded images.
        
        ### Features:
        - 🔬 **23 Disease Categories**: Comprehensive coverage of common skin conditions
        - 🎯 **High Accuracy**: Powered by ResNet50V2 transfer learning
        - 📱 **Easy to Use**: Simple drag-and-drop interface
        - 📊 **Detailed Analysis**: Confidence scores and probability distributions
        
        ### Supported Diseases:
        - Acne and Rosacea
        - Atopic Dermatitis
        - Bullous Disease
        - Cellulitis and Bacterial Infections
        - Eczema
        - Exanthems and Drug Eruptions
        - Hair Loss and Alopecia
        - Herpes and STDs
        - Light Diseases and Pigmentation Disorders
        - Lupus and Connective Tissue Diseases
        - Melanoma and Skin Cancer
        - Nail Fungus and Nail Diseases
        - Poison Ivy and Contact Dermatitis
        - Psoriasis and Lichen Planus
        - Scabies and Infestations
        - Seborrheic Keratoses and Benign Tumors
        - Systemic Disease
        - Tinea and Fungal Infections
        - Urticaria and Hives
        - Vascular Tumors
        - Vasculitis
        - Viral Infections (Warts, Molluscum)
        - Actinic Keratosis and Malignant Lesions
        
        ### How to Use:
        1. Go to "Upload & Predict" page
        2. Upload a clear image of the skin condition
        3. Get instant prediction with confidence score
        4. View detailed analysis and recommendations
        
        ⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.
        """)
        
    elif page == "Upload & Predict":
        st.header("📤 Upload Image for Diagnosis")
        
        # Load model
        detector = load_model()
        if detector is None:
            st.stop()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin condition"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("🔍 Analysis")
                
                # Preprocess image
                img_array = preprocess_image(image)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    prediction = detector.model.predict(img_array)
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class]
                    disease_name = detector.class_names[predicted_class]
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Disease:</h3>
                    <h2>{disease_name}</h2>
                    <p>Confidence: <span class="{get_confidence_color(confidence)}">{confidence:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence level indicator
                if confidence >= 0.8:
                    st.success("✅ High confidence prediction")
                elif confidence >= 0.6:
                    st.warning("⚠️ Medium confidence prediction")
                else:
                    st.error("❌ Low confidence prediction")
                
                # Top 5 predictions
                st.subheader("📊 Top 5 Predictions")
                top_5_indices = np.argsort(prediction[0])[-5:][::-1]
                
                for i, idx in enumerate(top_5_indices):
                    prob = prediction[0][idx]
                    disease = detector.class_names[idx]
                    st.write(f"{i+1}. {disease}: {prob:.2%}")
                
                # Bar chart of top predictions
                fig = go.Figure(data=[
                    go.Bar(
                        x=[detector.class_names[i] for i in top_5_indices],
                        y=[prediction[0][i] for i in top_5_indices],
                        marker_color=['#1f77b4' if i == predicted_class else '#ff7f0e' for i in top_5_indices]
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    xaxis_title="Disease",
                    yaxis_title="Probability",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if confidence >= 0.8:
                    st.info("""
                    **High confidence prediction detected.**
                    - Consider consulting a dermatologist for confirmation
                    - Monitor the condition for any changes
                    - Follow appropriate treatment guidelines
                    """)
                elif confidence >= 0.6:
                    st.warning("""
                    **Medium confidence prediction.**
                    - Multiple conditions may have similar symptoms
                    - Professional medical evaluation is recommended
                    - Consider additional diagnostic tests
                    """)
                else:
                    st.error("""
                    **Low confidence prediction.**
                    - Image quality or lighting may be affecting results
                    - Try uploading a clearer, better-lit image
                    - Professional medical evaluation is strongly recommended
                    """)
    
    elif page == "Model Information":
        st.header("🤖 Model Information")
        
        st.markdown("""
        ### Model Architecture
        
        **Base Model**: ResNet50V2 (Transfer Learning)
        - Pre-trained on ImageNet dataset
        - Excellent feature extraction capabilities
        - Proven performance on medical imaging tasks
        
        **Custom Layers**:
        - Global Average Pooling
        - Dropout layers (0.5, 0.3, 0.2) for regularization
        - Dense layers (512, 256, 23 units)
        - Softmax activation for multi-class classification
        
        ### Training Details
        
        **Data Augmentation**:
        - Rotation: ±20 degrees
        - Width/Height shift: ±20%
        - Shear: ±20%
        - Zoom: ±20%
        - Horizontal flip: Enabled
        
        **Training Strategy**:
        - Two-phase training: Feature extraction + Fine-tuning
        - Early stopping to prevent overfitting
        - Learning rate reduction on plateau
        - Model checkpointing for best weights
        
        **Dataset**:
        - 23 skin disease categories
        - Training/Validation split: 80/20
        - Separate test set for evaluation
        - Image size: 224x224 pixels
        
        ### Performance Metrics
        
        The model achieves high accuracy across multiple skin disease categories,
        with particular strength in distinguishing between common conditions
        like acne, eczema, and psoriasis.
        """)
        
        # Add model performance visualization if available
        if os.path.exists('training_history.png'):
            st.subheader("📈 Training History")
            st.image('training_history.png', caption="Model Training Progress")
        
        if os.path.exists('confusion_matrix.png'):
            st.subheader("📊 Confusion Matrix")
            st.image('confusion_matrix.png', caption="Model Performance by Disease Category")
    
    elif page == "About":
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### Purpose
        
        This Skin Disease Detection System is designed to assist in the preliminary
        identification of common skin conditions using artificial intelligence.
        
        ### Technology Stack
        
        - **Deep Learning**: TensorFlow 2.x with Keras
        - **Model**: ResNet50V2 with transfer learning
        - **Web Framework**: Streamlit
        - **Image Processing**: OpenCV, PIL
        - **Data Visualization**: Plotly, Matplotlib
        
        ### Dataset
        
        The model is trained on a comprehensive dataset containing images of
        23 different skin disease categories, carefully curated for medical
        imaging applications.
        
        ### Limitations
        
        - **Not a replacement for professional medical advice**
        - Accuracy may vary based on image quality and lighting
        - Some rare conditions may not be accurately classified
        - Results should be validated by healthcare professionals
        
        ### Privacy
        
        - Images are processed locally and not stored
        - No personal health information is collected
        - All predictions are temporary and not saved
        
        ### Future Improvements
        
        - Integration with electronic health records
        - Support for additional disease categories
        - Real-time video analysis capabilities
        - Mobile application development
        
        ### Contact
        
        For questions or feedback about this application, please contact me at maheshvellogi99@gmail.com\n
        Github: https://github.com/maheshvellogi99 \n
        Linkedin: https://www.linkedin.com/in/mahesh-vellogi-aa13522ba/
        
        """)

if __name__ == "__main__":
    main() 