import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from gtts import gTTS
import os

# Define the custom model loading function
def load_custom_model(model_path):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Load weights into the new model
    model.load_weights(model_path)
    return model   

# Load the model using the custom function
model = load_custom_model("fire_detection_model1.h5")

# Define the fire detection function
def detect_fire(frame, threshold=0.5):
    preprocess_frame = cv2.cvtColor(cv2.resize(frame, (48,48)), cv2.COLOR_BGR2GRAY)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=0)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=-1)
    preprocess_frame = preprocess_frame.astype("float32") / 255

    prediction = model.predict(preprocess_frame)
    if prediction[0][1] >= threshold:
        return True
    else:
        return False
    
# Open video capture
cap = cv2.VideoCapture("fire.mp4")
if not cap.isOpened():
    print("Error: could not open video file")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('finalvideo.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if detect_fire(frame):
        # Generate the alarm message
        alarm_message = "Warning! Fire is detected. Please evacuate immediately."
        
        # Create a gTTS object and save it as an MP3 file
        tts = gTTS(text=alarm_message, lang='en')
        tts.save("alarm.mp3")
        
        # Play the generated audio file
        os.system("start alarm.mp3")
        
        cv2.rectangle(frame, (100,100), (frame.shape[1]-100, frame.shape[0]-100), (0,0,255), 2)
        cv2.putText(frame, "Warning, fire is detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    # Write the frame into the output video
    out.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
