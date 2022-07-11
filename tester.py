import tensorflow.keras
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils


np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('arecanut_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

result = []
text = ""
foo = 1
while foo != 9:
    img = cv2.imread("dataset/Test/0/" + str(foo) + ".jpg")
    img = cv2.resize(img,(224, 224))

    image_array = np.asarray(img)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    for i in prediction:
        if i[0] > i[1]:
            text ="Bad"
        elif i[1] > i[0]:
            text ="Good"
        else:
            text = "Error"
        img = cv2.resize(img,(500, 500))
        cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    result.append(str(foo) + " - " + str(prediction) + " " + text)
    if key == ord("q"):
        break

    foo += 1
print("===================================")
print("-RESULT-")
print(*result, sep = "\n")


    # for i in prediction:
    #     if i[0] > 0.59:
    #         text ="Bad"
    #     if i[1] > 0.59:
    #         text ="Good"