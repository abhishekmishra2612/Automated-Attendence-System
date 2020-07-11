import numpy as np
import cv2
import face_recognition as fr
import AAS_Utility_functions

known_face_Encodings = AAS_Utility_functions.known_face_encodings
known_Rolls = AAS_Utility_functions.Rolls

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    Sframe = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    Sframe_rgb = cv2.cvtColor(Sframe, cv2.COLOR_BGR2RGB)

    try:
     Sframe_encode = fr.face_encodings(Sframe_rgb)[0]
     matches = fr.compare_faces(known_face_Encodings, Sframe_encode)
     face_Distances = fr.face_distance(known_face_Encodings, Sframe_encode)
     min_dist_index = np.argmin(face_Distances)

     rolls = []
     if matches[min_dist_index]:
         rolls.append(known_Rolls[min_dist_index])
     if len(rolls) > 0:
         cv2.putText(frame, AAS_Utility_functions.roll_to_name[rolls[0]], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
         AAS_Utility_functions.mark_Attendence(rolls[0])
     else:
         cv2.putText(frame, "Unknown", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except:
     cv2.putText(frame, "Face not found", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    finally:
     cv2.imshow("Web Cam", frame)

    if cv2.waitKey(10) == ord('q'): # wait until 'q' key is pressed
     break

cap.release()
cv2.destroyAllWindows()