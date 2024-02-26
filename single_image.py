from PIL import Image, ImageDraw
from imageai.Detection import ObjectDetection
import os

def crop_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    left = width // 2
    top = 0
    right = width
    bottom = height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def detect_objects_in_image(image_path):
    # Создаем экземпляр класса ObjectDetection и загружаем модель YOLOv3
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("yolov3.pt")
    detector.loadModel()

    custom = detector.CustomObjects(car=True)

    # Обрезаем изображение
    cropped_image = crop_image(image_path)

    # Детектируем объекты на обрезанном изображении
    detections = detector.detectObjectsFromImage(custom_objects=custom, input_image=cropped_image, minimum_percentage_probability=5)

    # Создаем объект ImageDraw для наложения рамок на изображение
    draw = ImageDraw.Draw(cropped_image)

    # Налагаем рамки на изображение в соответствии с результатами детекции
    for eachObject in detections:
        box = eachObject["box_points"]
        draw.rectangle(box, outline="red")

    # Показываем изображение с наложенными рамками
    cropped_image.show()

# Путь к изображению для обработки
image_path = input("Введите путь к изображению: ")

# Обрабатываем изображение и отображаем результаты детекции
detect_objects_in_image(image_path)
