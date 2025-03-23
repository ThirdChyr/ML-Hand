import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import paho.mqtt.client as mqtt  


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


cap = cv2.VideoCapture(2)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'Good', 1: 'Bad', 2: 'Hello', 3: 'Repeat', 4: 'I', 5: 'Non'}

current_class = None
class_start_time = None
last_detected_time = time.time()
detected_classes = []
cooldown_start_time = None

MQTT_BROKER = "20.243.148.107"  
MQTT_PORT = 1883
MQTT_TOPIC = "Translate/Massage"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

last_detected_class = None

def process_frame(frame, detected_classes, current_class, class_start_time):
   
    global last_detected_time, last_detected_class, cooldown_start_time
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

            
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

           
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            
            if current_class == predicted_character:
                if class_start_time is None:
                    class_start_time = time.time()
                elif time.time() - class_start_time >= 2:  
                    if cooldown_start_time is None or time.time() - cooldown_start_time >= 2:  
                        detected_classes.append(predicted_character)
                        last_detected_class = predicted_character  
                        cooldown_start_time = time.time()  
                        print(f"Class '{predicted_character}' detected and saved!")

            else:
                current_class = predicted_character
                class_start_time = time.time()

            last_detected_time = time.time()

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    else:
       
        last_detected_class = None

    return frame, detected_classes, current_class, class_start_time

def draw_results(frame, detected_classes):
    H, W, _ = frame.shape
    detected_text = f"You Text is: {' '.join(detected_classes)}"
    cv2.rectangle(frame, (10, H - 50), (W - 10, H - 10), (255, 255, 255), -1)
    cv2.putText(frame, detected_text, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

def reset_detected_classes():
    global detected_classes
    detected_classes = []

def main():
    global current_class, class_start_time, last_detected_time, detected_classes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame, detected_classes, current_class, class_start_time = process_frame(frame, detected_classes, current_class, class_start_time)

        
        if time.time() - last_detected_time >= 5 and detected_classes:
            
            message = f"{len(detected_classes)}{''.join(detected_classes)}"
            mqtt_client.publish(MQTT_TOPIC, message)  
            print(f"Sent MQTT message: {message}")

            
            reset_detected_classes()

  
        frame = draw_results(frame, detected_classes)

        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()