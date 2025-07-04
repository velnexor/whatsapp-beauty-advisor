import cv2
import numpy as np
import tensorflow as tf

def predict_face_shape_from_array(image_array):
    model = tf.keras.models.load_model('models/face_shape_model.h5')
    img = cv2.resize(image_array, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    FACE_SHAPES = ['oval', 'round', 'square']
    return FACE_SHAPES[predicted_class], confidence

def main():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture an image, or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Webcam - Press SPACE to scan', frame)
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            print('Escape hit, closing...')
            break
        elif k%256 == 32:  # SPACE pressed
            face_shape, confidence = predict_face_shape_from_array(frame)
            print(f"Predicted Face Shape: {face_shape} (Confidence: {confidence:.2f})")
            cv2.putText(frame, f"{face_shape} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Result', frame)
            cv2.waitKey(3000)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
