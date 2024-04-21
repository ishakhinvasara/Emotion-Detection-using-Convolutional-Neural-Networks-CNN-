
# pip install opencv-python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
def take_photo(filename='photo.jpg', quality=0.8):
js = Javascript('''
async function takePhoto(quality) {
const div = document.createElement('div');
const capture = document.createElement('button');
capture.textContent = 'Capture';
div.appendChild(capture);
const video = document.createElement('video');
video.style.display = 'block';
const stream = await navigator.mediaDevices.getUserMedia({video:
true});
document.body.appendChild(div);
div.appendChild(video);
video.srcObject = stream;
await video.play();

// Resize the output to fit the video element.
google.colab.output.setIframeHeight(document.documentElement.scrollHeight,
true);
// Wait for Capture to be clicked.
await new Promise((resolve) => capture.onclick = resolve);
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);
stream.getVideoTracks()[0].stop();
div.remove();
return canvas.toDataURL('image/jpeg', quality);
}
''')
display(js)
data = eval_js('takePhoto({})'.format(quality))
binary = b64decode(data.split(',')[1])
with open(filename, 'wb') as f:
f.write(binary)
return filename
from IPython.display import Image
try:
filename = take_photo()
print('Saved to {}'.format(filename))
# Show the image which was just taken.
display(Image(filename))
except Exception as err:
# Errors will be thrown if the user does not have a webcam or if they do
not
# grant the page permission to access it.
print(str(err))


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
# Define the directories for train and test data
train_dir = '/content/drive/MyDrive/Colab Notebooks/new dataset/archive
(7)/train'
test_dir = '/content/drive/MyDrive/Colab Notebooks/new dataset/archive
(7)/test'
# Get the list of emotions from the train directory
emotions = os.listdir(train_dir)
print("\n ",emotions,"\n")

X_train = []
y_train = []
X_test = []
y_test = []

# Load training images
for emotion in emotions:
train_images = os.listdir(os.path.join(train_dir, emotion))
for image in train_images:
img_path = os.path.join(train_dir, emotion, image)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
X_train.append(img)
y_train.append(emotions.index(emotion))
else:
print(f"Skipping {img_path} as it could not be loaded.")
# Load testing images
for emotion in emotions:
test_images = os.listdir(os.path.join(test_dir, emotion))
for image in test_images:
img_path = os.path.join(test_dir, emotion, image)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
X_test.append(img)
y_test.append(emotions.index(emotion))
else:
print(f"Skipping {img_path} as it could not be loaded.")
# Convert lists to numpy arrays and normalize pixel values
X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)
# Define CNN Model
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(X_train.shape[1], X_train.shape[2], 1)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(len(emotions), activation='softmax')
])

# Train the Model
history = model.fit(X_train.reshape(-1, X_train.shape[1],
X_train.shape[2], 1), y_train,
epochs=14, batch_size=32, validation_split=0.2)
# Evaluate the Model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test.reshape(-1, X_train.shape[1],
X_train.shape[2], 1), y_test)
print("Test Accuracy:", accuracy)


#Plotting Accuracy & Loss
plt.style.use('dark_background')
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')
plt.show()

from google.colab.patches import cv2_imshow
# Function to detect emotion from an image
def detect_emotion_from_image(model, emotions, image_path):
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
# Read the image
frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
face_img = gray[y:y+h, x:x+w]
resized = cv2.resize(face_img, (48, 48))
normalized = resized / 255.0
reshaped = normalized.reshape(1, 48, 48, 1)
result = model.predict(reshaped)
label = np.argmax(result)
emotion = emotions[label]
cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
0.9, (255, 0, 0), 2, cv2.LINE_AA)
cv2_imshow(frame)
return emotion

# Call the function to detect emotion from the captured image

image_path = '/content/photo.jpg' # Replace with the path to your test
image
emotiondetected = detect_emotion_from_image(model, emotions, image_path)
print("\n",emotiondetected)


neutral
if(emotiondetected=='neutral' or emotiondetected=='surprised' ):
emotiondetected='happy'
if(emotiondetected=='angry'or emotiondetected=='disgusted'):
emotiondetected='calm'
