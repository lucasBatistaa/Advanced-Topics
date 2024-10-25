import cv2 as cv
import numpy as np

import tensorflow as tf

# Load the pre-trained Teachable Machine model
model = tf.keras.models.load_model('data/fall_detection_model.h5')

# Initialize webcam
cap = cv.VideoCapture(0)

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv.resize(frame, (224, 224))  # Resize to the input size of the model
    frame = np.array(frame, dtype=np.float32)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model prediction
    input_frame = preprocess_frame(frame)

    # Predict using the model
    predictions = model.predict(input_frame)
    fall_detected = np.argmax(predictions[0])  # Get the index of the highest prediction
    
    # 0 = no fall, 1 = fall (assuming the model is trained with these labels)
    if fall_detected == 1:
        cv.putText(frame, 'Fall Detected!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame, 'No Fall', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Display the video
    cv.imshow('Fall Detection', frame)

    # Break the loop with 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()