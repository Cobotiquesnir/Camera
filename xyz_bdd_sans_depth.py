import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import mysql.connector

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configurer la connexion à la base de données MySQL
cnx = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="xyz"
)
cursor = cnx.cursor()

# Configurer la caméra Intel RealSense D415
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialiser MediaPipe Hands
with mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while True:
        #permet de réenitialiser le compte "id" (recommence les "id" à 0)
        cursor.execute("ALTER TABLE co AUTO_INCREMENT = 1")
        cnx.commit()

        # Obtenir les images de la caméra et aligner la profondeur sur la couleur
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        # Traiter l'image avec MediaPipe Hands
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        
        # Dessiner les résultats
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Insérer les coordonnées x, y et z pour chaque point de repère
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    
                    if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                        z = depth_frame.get_distance(x, y)
                        print(f"Landmark {id}: x={x}, y={y}, z={z}")
                        
                        # Insérer les coordonnées x, y et z dans la base de données avec un ID unique
                        insert_query = """INSERT INTO co (x, y, z) VALUES (%s, %s, %s)"""

                        cursor.execute(insert_query, (x, y, z))
                        cnx.commit()
        
        # Afficher l'image couleur
        cv2.imshow("Color Frame", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

pipeline.stop()
cv2.destroyAllWindows()
