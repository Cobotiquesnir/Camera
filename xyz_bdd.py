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
        # Obtenir les images de la caméra et aligner la profondeur sur la couleur
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
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
                
                # Afficher les coordonnées x, y et z pour chaque point de repère
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    
                    if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                        z = depth_frame.get_distance(x, y)
                        print(f"Landmark {id}: x={x}, y={y}, z={z}")
                        
                        # Insérer les coordonnées x, y et z dans la base de données
                        #insert_query = "INSERT INTO co (id, x, y, z) VALUES (%s, %s, %s, %s)"
                        insert_query = """INSERT INTO co (id, x, y, z) VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE x=VALUES(x), y=VALUES(y), z=VALUES(z)"""

                        cursor.execute(insert_query, (id, x, y, z))
                        cnx.commit()
        
        # Afficher les images
        cv2.imshow("Color Frame", image)
        cv2.imshow("Depth Frame", depth_colormap)

        if cv2.waitKey(5) & 0xFF == 27:
            break

pipeline.stop()
cv2.destroyAllWindows()
