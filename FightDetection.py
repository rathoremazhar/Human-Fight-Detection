import copy
import threading
import os
import cv2
import numpy as np
from datetime import datetime
import time
from tensorflow import keras
from playsound3 import playsound
from PIL import Image

# Parameters
text = "Fight frame"
count = 0
fs = 0
IMG_SIZE = 128
MAX_SEQ_LENGTH = 42
NUM_FEATURES = 2048

# Load model
filepath = "cnn_rnn_model/cnn_rnn_model-2.h5"

def play_alarm_sound_async():
    sound_file_path = "sound.wav"
    playsound(sound_file_path)

def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.LSTM(32, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.LSTM(16, return_sequences=True)(x)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(2, activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_cross entropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return rnn_model

# Load model weights
seq_model = get_sequence_model()
seq_model.load_weights(filepath)
print("✅ Model weights loaded.")

# Build feature extractor
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# -------------------- PREDICT VIDEO FROM FILE --------------------

def predict_from_video_file(video_path):
    global count, fs


    print(f"📂 Processing video: {video_path}")
    vid = cv2.VideoCapture(video_path)

    original_fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Original video FPS: {original_fps:.2f}")

    process_every_nth = max(int(original_fps / 5), 1)  # Ensure at least 1
    print(f"⚙️  Skipping frames to process at ~5 FPS (every {process_every_nth}th frame)...\n")

    num_samples = 1
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    frame_counter = 0
    processed_frames = 0
    total_start_time = time.time()  # 🟢 Start timing

    fight_fram_counter = 0
    nofight_fram_counter = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        frame_counter += 1  # 🟢 Count frame

        # Skip frames to simulate 5 FPS
        if frame_counter % process_every_nth != 0:
            continue

        processed_frames += 1

        frame_cp = copy.deepcopy(frame)
        frame_cp = cv2.resize(frame_cp, (IMG_SIZE, IMG_SIZE))
        start_time = time.time()

        frame_features[:, 0:-2, :] = frame_features[:, 1:-1, :]
        frame_features[:, -1, :] = feature_extractor.predict(
            frame_cp.reshape(1, IMG_SIZE, IMG_SIZE, -1)
        )

        frame_masks[:, :MAX_SEQ_LENGTH] = 1
        prediction = seq_model.predict([frame_features, frame_masks])
        end_time = time.time()

        prediction = prediction.argmax()
        CLASSES = ["nofight", "fight"]
        label = CLASSES[prediction]

        print(f"Prediction: {label.upper()}, Time per frame: {end_time - start_time:.2f}s")
        if label == "fight":
            fight_fram_counter += 1
            print(f"🟥 FIGHT Frame counter: {fight_fram_counter},  and overall video frame number: {frame_counter}")
        elif label == "nofight":
            nofight_fram_counter += 1
            print(f"🟥 No FIGHT Frame counter: {nofight_fram_counter},  and overall video frame number: {frame_counter}")

        if fight_fram_counter >= 10:
            print("⚠️ FIGHT DETECTED")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame_rgb)
            filename = f'fight_data/{datetime.now().date()}-{datetime.now().strftime("%H")}-{datetime.now().strftime("%M")}-{fs}.jpg'
            os.makedirs("fight_data", exist_ok=True)
            im.save(filename)
            fs += 1 #did not understand why we are using this
            threading.Thread(target=play_alarm_sound_async).start()
            fight_fram_counter = 0  # reinitialize fight frame counter and restart fight detection.
            nofight_fram_counter=0 # also reinitialize non-fight frame counter and restart fight detection.

        elif nofight_fram_counter >= 5:
            print("🔁 Resetting fight counter (no fight frames observed)")
            fight_fram_counter = 0  # reinitialize fight frame counter and restart fight detection.
            nofight_fram_counter = 0  # also reinitialize non-fight frame counter and restart fight detection.

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_end_time = time.time()  # 🟢 End timing
    total_duration = total_end_time - total_start_time
    vid.release()
    print("\n✅ Video processing complete.")
    print(f"📊 Total frames read: {frame_counter}")
    print(f"📊 Total frames processed (~5 FPS): {processed_frames}")
    print(f"⏱️ Total time taken: {total_duration:.2f} seconds")

# ------------------------ MAIN ------------------------
#########Run this main function only when you don't have a main function in nay other file.
# if __name__ == "__main__":
#     print("=== Fight Detection & Alert System (Terminal Mode) ===")
#
#     while True:
#         video_path = input("\n📹 Enter path to video file (or type 'exit' to quit): ").strip()
#
#         if video_path.lower() == 'exit':
#             print("👋 Exiting program. Goodbye!")
#             break
#
#         if not os.path.isfile(video_path):
#             print("❌ Invalid file path. Please try again.")
#             continue
#
#         predict_from_video_file(video_path)
#
#         again = input("\n🔁 Do you want to process another video? (y/n): ").strip().lower()
#         if again not in ['y', 'yes']:
#             print("👋 Exiting program. Goodbye!")
#             break
