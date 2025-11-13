import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

print("[INFO] Starting Emotion Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        
        cv2.putText(frame, f"{emotion} ({confidence:.2f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)

    
    cv2.imshow('Emotion Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
