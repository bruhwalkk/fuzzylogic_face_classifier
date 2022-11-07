from PIL import Image, ImageDraw, ImageFont
import cv2
import dlib
import math
import os
import pandas as pd
import matlab.engine


def get_face(img):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point
        crop = img[y1:y2, x1:x2]
    return crop

def get_dominant_color(pil_img, palette_size=32):
    img = pil_img.copy()
    img.thumbnail((300, 300))

    paletted = img.convert('P', colors=palette_size)

    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]

    return dominant_color

def get_color(img):
    image = Image.fromarray(img).convert("L")
    return float(round(100 * get_dominant_color(image)[0] / 255, 2))

def distance(c, dot1, dot2):
    return math.hypot(c[dot2][0] - c[dot1][0], c[dot2][1] - c[dot1][1])

def get_points(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    coords = []
    for face in faces:
        landmarks = predictor(image=gray, box=face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            coords.append((x,y))
    return coords

def get_eye_size(points):
    return round((distance(points, 37, 41) + distance(points, 38, 40) + distance(points, 43, 47) + distance(points, 44, 46)) / 4, 2)

def get_between_eyes_size(points):
    return round(distance(points, 42, 39), 2)

def get_nose_size(points):
    return round(distance(points, 33, 27), 2)

def get_nose_square(points):
    return round((distance(points, 33, 27) + distance(points, 35, 31)) * 0.5, 2)

def get_lips_size(points):
    return round((distance(points, 58, 50) + distance(points, 57, 51) + distance(points, 56, 52)) / 3, 2)

def get_lips_square(points):
    return round((distance(points, 54, 48) + distance(points, 57, 51)) * 0.5, 2)

def inspect_image(img_path, fis):
    l = []
    l.append(img_path)
    image = cv2.resize(get_face(cv2.imread(img_path)), (500,500), interpolation = cv2.INTER_AREA)
    l.append(get_color(image))
    points = get_points(image)
    l.append(get_eye_size(points))
    l.append(get_lips_size(points))
    l.append(get_between_eyes_size(points))
    #l.append(get_nose_size(points))
    #l.append(get_nose_square(points))
    #l.append(get_lips_square(points))
    return l

eng = matlab.engine.start_matlab()
fis = eng.readfis('mamdanitype1.fis')
folder_dir = "pics2"
images = []
results = []
for image in os.listdir(folder_dir):
    if (image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")):
        images.append(image)
for i in images:
    results.append(inspect_image(folder_dir + "\\" + i, fis))
df = pd.DataFrame(results, columns = ["name", "skin_lightness", "eye_size","lips_size", "between_eyes_size"])
print(df)
df.pop("name")
df.to_csv('C:\\fuzzy\\results_dataset.csv', sep = ",", header = False, index = False)

M = eng.csvread("results_dataset.csv")
for i in range(len(M)):
    output = eng.evalfis(fis, M[i])
    if output < 0.33:
        race = "asian"
    elif (output < 0.66) and (output > 0.33):
        race = "white"
    elif output > 0.66:
        race = "black"
    img = Image.open(folder_dir + "\\" + images[i])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Gidole-Regular.ttf", size = 100)
    draw.text((0,0), race, font = font, fill = "red")
    img.show()








