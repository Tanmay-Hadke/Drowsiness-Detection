# Driver's Drowsiness Detection

## Overview  
This project implements a **Driver's Drowsiness Detection System** using deep learning and computer vision techniques. The system monitors the driver in real-time and detects signs of drowsiness by analyzing facial features, focusing on eye behavior. 

Key features:  
- Real-time face and eye detection using Haar Cascade Classifiers.  
- Deep learning-based model (`cnn.h5`) for detecting drowsiness states.  
- Implementation provided in Jupyter notebooks for training and testing.  

---

## File Structure  

- **`cnn.h5`**  
  The pre-trained Convolutional Neural Network (CNN) model for classifying drowsy vs. non-drowsy states.  

- **Jupyter Notebooks**:  
  - `drowsiness detection.ipynb` - Main notebook for implementing real-time detection.  
  - `drowsiness_detection_training_notebook.ipynb` - Notebook for training the CNN model.  

- **Haar Cascade XML Files**:  
  - `haarcascade_frontalface_alt.xml` - For detecting faces in real-time.  
  - `haarcascade_lefteye_2splits.xml` and `haarcascade_righteye_2splits.xml` - For detecting left and right eyes.  

---

## Requirements  

- **Languages & Frameworks**:  
  - Python 3.7+  
  - TensorFlow/Keras  
  - OpenCV  

- **Python Libraries**:  
  ```bash
  pip install numpy opencv-python keras tensorflow
  ```

---

## How to Run  

1. **Clone this Repository**  
   ```bash
   git clone https://github.com/Tanmay-Hadke/Drowsiness-Detection/
   cd Drowsiness-Detection
   ```

2. **Prepare Environment**  
   Install the required dependencies as mentioned above.  

3. **Run the Real-Time Detection Notebook**  
   Open `drowsiness detection.ipynb` in Jupyter Notebook or a similar environment and execute the cells. Ensure your webcam is accessible.  

4. **Training the Model (Optional)**  
   If you'd like to retrain the model, open `drowsiness_detection_training_notebook.ipynb` and follow the instructions provided.

---

## Project Workflow  

1. **Face and Eye Detection**:  
   The Haar Cascade classifiers identify the face and eyes in the video stream.  

2. **Feature Extraction**:  
   Eye regions are analyzed to determine whether the eyes are open or closed.  

3. **Drowsiness Classification**:  
   The CNN model processes extracted features and classifies the driver's state as drowsy or alert.  

4. **Alert System**:  
   If the driver is detected as drowsy, an alert mechanism (e.g., sound) is triggered.  

---

## Future Enhancements  

- Incorporating head posture analysis for improved accuracy.  
- Deploying on embedded systems for real-world use.  
- Adding multi-modal detection using heart rate or yawning frequency.  

---

## Acknowledgments  

- **Haar Cascade Classifiers**: OpenCV pre-trained models.  
- Deep Learning Framework: TensorFlow/Keras.  

Feel free to contribute by submitting issues or pull requests!  

--- 
