#  Automated Road Damage Classification using Deep Learning

##  Project Overview

This project is a deep learning-based web application that automatically detects and classifies road damage from images. It helps in identifying common road issues such as potholes, cracks, and manholes using computer vision techniques.

The system is designed to support smart city infrastructure by enabling faster and more accurate road condition monitoring.


##  Problem Statement

Manual road inspection is time-consuming, expensive, and prone to human error. Poor road conditions can lead to accidents and vehicle damage.
This project aims to automate road damage detection using deep learning models and provide real-time predictions through a web application.


## Solution

* Built a deep learning model using **Transfer Learning**
* Classified road images into:

  * Pothole
  * Crack
  * Manhole
* Developed a **Streamlit web application** for real-time predictions
* Provided confidence scores and repair recommendations

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Streamlit
* Matplotlib / Seaborn


## Project Structure

Road_Damage_App/
│
├── data/                         # Dataset (train/test images)
├── road_damage_model_final.keras # Trained model
├── app.py                        # Streamlit web app
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation


## How to Run the Project

### 1 Clone the Repository

```
git clone https://github.com/your-username/road-damage-classification.git
cd road-damage-classification
```

### 2 Install Dependencies

```
pip install -r requirements.txt
```

### 3 Run the Application

```
streamlit run app.py
```

### 4 Open in Browser

```
http://localhost:8501
```


## Model Details

* Model Type: Transfer Learning (EfficientNet / MobileNet / ResNet)
* Input Size: 224x224 images
* Output Classes: 3 (Pothole, Crack, Manhole)

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


## Features
* Upload road images
* Real-time damage classification
* Confidence score display
* Damage-specific recommendations
* Lightweight and user-friendly interface

## Sample Output
* **Prediction:** Pothole
* **Confidence:** 92.45%
* **Recommendation:** Immediate repair required

## Use Cases
* Smart City Monitoring
* Municipal Road Maintenance
* Transportation Safety Analysis
* Public Issue Reporting Systems

## Future Improvements
* Add Grad-CAM visualization for explainability
* Deploy application to cloud (Streamlit Cloud / Render)
* Improve model accuracy with larger dataset
* Add real-time video detection

## Author
Sakthi

## Acknowledgements
* Road Damage Dataset (RDD2020 / RDD2022)
* TensorFlow & Keras Documentation
* Streamlit Community

## Contact
Feel free to connect for collaboration or feedback!
