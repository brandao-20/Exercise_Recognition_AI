import streamlit as st
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

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Suprimir avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Função para bloco de atenção
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# Construção e carregamento do modelo
@st.cache_resource
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

# Interface do Streamlit
st.write("# AI Personal Fitness Trainer Web App")
st.write("## Configurações")
threshold1 = st.slider("Confiança Mínima de Detecção de Keypoints", 0.00, 1.00, 0.30)
threshold2 = st.slider("Confiança Mínima de Rastreamento", 0.00, 1.00, 0.30)
threshold3 = st.slider("Confiança Mínima de Classificação de Atividade", 0.00, 1.00, 0.20)

st.write("## Limiares para Agachamento")
thr_down = st.slider("Limiar para 'down' (graus)", 90, 150, 110)  # Ajustado para 110°
thr_up = st.slider("Limiar para 'up' (graus)", 130, 180, 160)

st.write("## Selecionar Vídeo para Processar")
video_option = st.selectbox("Escolha um vídeo", ["Execução Correta", "Execução Incorreta"])
video_path = {
    "Execução Correta": "D:\\uni\\Exercise_Recognition_AI\\videos\\correto.mp4",
    "Execução Incorreta": "D:\\uni\\Exercise_Recognition_AI\\videos\\exercicio2.mp4"
}[video_option]

# Configuração do MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2)

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

# Classe para processamento de vídeo
class VideoProcessor:
    def __init__(self, thr_down=110, thr_up=160):  # Ajuste nos limiares
        self.actions = np.array(['curl', 'press', 'agachamento', 'flexão'])  # Termos em português
        self.sequence_length = 30
        self.threshold = threshold3
        self.sequence = []
        self.current_action = ''
        self.action_history = deque(maxlen=15)
        self.squat_counter = 0
        self.squat_stage = None
        self.thr_down = thr_down
        self.thr_up = thr_up
        self.feedback_timer = 0
        self.feedback_message = ""
        self.feedback_color = (255, 255, 255)

    def mediapipe_detection(self, image, model):
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = model.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr, results, (width, height)

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def extract_keypoints(self, results):
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*4)
        return pose

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle

    def get_coordinates(self, landmarks, mp_pose, side, joint):
        coord = getattr(mp_pose.PoseLandmark, side.upper() + "_" + joint.upper())
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        return [x_coord_val, y_coord_val]

    def viz_joint_angle(self, image, angle, joint, width, height):
        pos = tuple(np.multiply(joint, [width, height]).astype(int))
        image = draw_text_pil(image, str(int(angle)), pos, font_size=15, color=(255, 255, 255), bg_color=(50, 50, 50))

    def count_reps(self, image, current_action, landmarks, mp_pose, width, height):
        debug_info = {
            "squat_stage": self.squat_stage,
            "message": "",
            "quality_feedback": "",
            "status": ""
        }

        if current_action == 'agachamento' and landmarks:
            left_hip = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
            left_knee = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
            left_ankle = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            right_hip = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
            right_knee = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
            right_ankle = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')

            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            logging.debug(f"Ângulo joelho esquerdo: {left_knee_angle}, Ângulo joelho direito: {right_knee_angle}")

            if left_knee_angle < self.thr_down and right_knee_angle < self.thr_down:
                self.squat_stage = "down"
                debug_info["squat_stage"] = "down"
                debug_info["quality_feedback"] = "Boa profundidade! Suba para completar."
            elif left_knee_angle > self.thr_up and right_knee_angle > self.thr_up and self.squat_stage == "down":
                self.squat_stage = "up"
                self.squat_counter += 1
                debug_info["message"] = f"Agachamento #{self.squat_counter} concluído!"
                debug_info["status"] = "Agachamento bem executado!"
                debug_info["quality_feedback"] = "Ótima forma!"
            else:
                if self.squat_stage == "down":
                    debug_info["quality_feedback"] = "Suba para completar o agachamento!"
                else:
                    debug_info["quality_feedback"] = "Abaixe mais os quadris!"

            self.viz_joint_angle(image, left_knee_angle, left_knee, width, height)
            self.viz_joint_angle(image, right_knee_angle, right_knee, width, height)

        return debug_info

    def process(self, image):
        image, results, (width, height) = self.mediapipe_detection(image, pose)
        self.draw_landmarks(image, results)

        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        self.sequence = self.sequence[-self.sequence_length:]

        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            predicted_action = self.actions[np.argmax(res)]
            confidence = np.max(res)
            self.action_history.append(predicted_action)
            self.current_action = max(set(self.action_history), key=self.action_history.count)
            if confidence < self.threshold:
                self.current_action = ''
            logging.debug(f"Predição: {predicted_action}, Confiança: {confidence:.2f}, Ação Atual: {self.current_action}")

        debug_info = {
            "squat_stage": None,
            "message": "",
            "quality_feedback": "",
            "status": ""
        }

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                debug_info = self.count_reps(image, self.current_action, landmarks, mp_pose, width, height)
            except Exception as e:
                logging.error(f"Erro ao processar landmarks: {e}")
                debug_info["message"] = "Erro: Landmarks não detectados"
        else:
            logging.warning("Nenhuma pose detectada.")
            debug_info["message"] = "Nenhuma pose detectada"

        y_offset = 30

        if self.current_action:
            action_display = "Agachamento" if self.current_action == "agachamento" else self.current_action
            text = f"Ação: {action_display} (Confiança: {confidence:.2f})"
            image = draw_text_pil(image, text, (10, y_offset), font_size=30, color=(0, 255, 0), bg_color=(50, 50, 50))
            y_offset += 40

        text = f"Agachamento: {self.squat_counter}"
        image = draw_text_pil(image, text, (10, y_offset), font_size=30, color=(255, 255, 0), bg_color=(50, 50, 50))
        y_offset += 40

        if self.feedback_timer > 0:
            image = draw_text_pil(image, self.feedback_message, (10, y_offset), font_size=30, color=self.feedback_color, bg_color=(50, 50, 50))
            self.feedback_timer -= 1
        else:
            if debug_info["quality_feedback"]:
                self.feedback_message = debug_info["quality_feedback"]
                self.feedback_timer = 30  # 1 segundo a 30 FPS
                self.feedback_color = (255, 0, 0)  # Vermelho
            elif debug_info["status"]:
                self.feedback_message = debug_info["status"]
                self.feedback_timer = 30
                self.feedback_color = (255, 255, 255)  # Branco
            elif debug_info["message"]:
                self.feedback_message = debug_info["message"]
                self.feedback_timer = 30
                self.feedback_color = (0, 0, 255)  # Azul

        return image

# Processamento do vídeo
processor = VideoProcessor(thr_down=thr_down, thr_up=thr_up)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("Erro ao abrir o vídeo.")
else:
    st.write("Processando o vídeo...")
    frame_placeholder = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Fim do vídeo.")
            break
        image = processor.process(frame)
        frame_placeholder.image(image, channels="BGR")
    cap.release()