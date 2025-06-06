import cv2
import numpy as np
import mediapipe as mp
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
from collections import deque
import os
import logging
from PIL import Image, ImageDraw, ImageFont

# Suprimir avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Função para bloco de atenção
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# Construção e carregamento do modelo
def build_model(HIDDEN_UNITS=128, sequence_length=30, num_input_values=33*4, num_classes=4):
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)
    load_dir = "./LSTM_Attention_128HUs.h5"
    model.load_weights(load_dir)
    return model

HIDDEN_UNITS = 128
model = build_model(HIDDEN_UNITS)

# Configuração do MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Funções auxiliares
def mediapipe_detection(image, model):
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results, (width, height)

def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    return pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def get_coordinates(landmarks, mp_pose, side, joint):
    coord = getattr(mp_pose.PoseLandmark, side.upper() + "_" + joint.upper())
    x_coord_val = landmarks[coord.value].x
    y_coord_val = landmarks[coord.value].y
    return [x_coord_val, y_coord_val]

def viz_joint_angle(image, angle, joint, width, height):
    pos = tuple(np.multiply(joint, [width, height]).astype(int))
    image = draw_text_pil(image, str(int(angle)), pos, font_size=15, color=(255, 255, 255), bg_color=(50, 50, 50))
    return image

# Função para desenhar texto com fundo usando Pillow
def draw_text_pil(image, text, position, font_size=30, color=(255, 255, 255), bg_color=(50, 50, 50)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(text_bbox, fill=bg_color)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Variáveis globais
sequence = []
actions = np.array(['curl', 'press', 'agachamento', 'flexão'])
threshold = 0.5
sequence_length = 30
squat_counter = 0
squat_stage = None
action_history = deque(maxlen=15)
feedback_timer = 0
feedback_message = ""
feedback_color = (255, 255, 255)
thr_down = 110  # Limiar ajustado para 110°
thr_up = 160

# Função de contagem de repetições para agachamento
def count_reps(image, current_action, landmarks, mp_pose, width, height):
    global squat_counter, squat_stage, feedback_timer, feedback_message, feedback_color

    debug_info = {
        "squat_stage": squat_stage,
        "message": "",
        "quality_feedback": "",
        "status": ""
    }

    if current_action == 'agachamento' and landmarks:
        left_hip = get_coordinates(landmarks, mp_pose, 'left', 'hip')
        left_knee = get_coordinates(landmarks, mp_pose, 'left', 'knee')
        left_ankle = get_coordinates(landmarks, mp_pose, 'left', 'ankle')
        right_hip = get_coordinates(landmarks, mp_pose, 'right', 'hip')
        right_knee = get_coordinates(landmarks, mp_pose, 'right', 'knee')
        right_ankle = get_coordinates(landmarks, mp_pose, 'right', 'ankle')

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        logging.debug(f"Ângulo joelho esquerdo: {left_knee_angle}, Ângulo joelho direito: {right_knee_angle}")

        if left_knee_angle < thr_down and right_knee_angle < thr_down:
            squat_stage = "down"
            debug_info["squat_stage"] = "down"
            debug_info["quality_feedback"] = "Boa profundidade! Suba para completar."
        elif left_knee_angle > thr_up and right_knee_angle > thr_up and squat_stage == "down":
            squat_stage = "up"
            squat_counter += 1
            debug_info["message"] = f"Agachamento #{squat_counter} concluído!"
            debug_info["status"] = "Agachamento bem executado!"
            debug_info["quality_feedback"] = "Ótima forma!"
        else:
            if squat_stage == "down":
                debug_info["quality_feedback"] = "Suba para completar o agachamento!"
            else:
                debug_info["quality_feedback"] = "Abaixe mais os quadris!"

        image = viz_joint_angle(image, left_knee_angle, left_knee, width, height)
        image = viz_joint_angle(image, right_knee_angle, right_knee, width, height)

    return debug_info, image

# Seleção de vídeo
print("Escolha o vídeo a processar:")
print("1 - Execução Correta")
print("2 - Execução Incorreta")
choice = input("Digite 1 ou 2: ")
video_path = "D:\\uni\\Exercise_Recognition_AI\\videos\\correto.mp4" if choice == '1' else "D:\\uni\\Exercise_Recognition_AI\\videos\\incorreto.mp4"

# Processamento do vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    image, results, (width, height) = mediapipe_detection(frame, pose)
    draw_landmarks(image, results)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    current_action = ''
    confidence = 0.0
    if len(sequence) == sequence_length:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_action = actions[np.argmax(res)]
        confidence = np.max(res)
        action_history.append(predicted_action)
        current_action = max(set(action_history), key=action_history.count)
        if confidence < threshold:
            current_action = ''

    debug_info = {
        "squat_stage": None,
        "message": "",
        "quality_feedback": "",
        "status": ""
    }

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            debug_info, image = count_reps(image, current_action, landmarks, mp_pose, width, height)
        except Exception as e:
            logging.error(f"Erro ao processar landmarks: {e}")
            debug_info["message"] = "Erro: Landmarks não detectados"
    else:
        logging.warning("Nenhuma pose detectada.")
        debug_info["message"] = "Nenhuma pose detectada"

    y_offset = 30

    if current_action:
        action_display = "Agachamento" if current_action == "agachamento" else current_action
        text = f"Ação: {action_display} (Confiança: {confidence:.2f})"
        image = draw_text_pil(image, text, (10, y_offset), font_size=30, color=(0, 255, 0), bg_color=(50, 50, 50))
        y_offset += 40

    text = f"Agachamento: {squat_counter}"
    image = draw_text_pil(image, text, (10, y_offset), font_size=30, color=(255, 255, 0), bg_color=(50, 50, 50))
    y_offset += 40

    if feedback_timer > 0:
        image = draw_text_pil(image, feedback_message, (10, y_offset), font_size=30, color=feedback_color, bg_color=(50, 50, 50))
        feedback_timer -= 1
    else:
        if debug_info["quality_feedback"]:
            feedback_message = debug_info["quality_feedback"]
            feedback_timer = 30  # 1 segundo a 30 FPS
            feedback_color = (255, 0, 0)  # Vermelho
        elif debug_info["status"]:
            feedback_message = debug_info["status"]
            feedback_timer = 30
            feedback_color = (255, 255, 255)  # Branco
        elif debug_info["message"]:
            feedback_message = debug_info["message"]
            feedback_timer = 30
            feedback_color = (0, 0, 255)  # Azul

    cv2.imshow('Video Feed', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()