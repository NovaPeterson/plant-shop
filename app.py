from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Load the model
def custom_loss_deserializer(config):
    return tf.keras.losses.deserialize({'class_name': 'SparseCategoricalCrossentropy', 'config': config})

model = tf.keras.models.load_model('./Plant_Island.h5', custom_objects={'SparseCategoricalCrossentropy': custom_loss_deserializer})

# Define class names
class_names = ['Common Lanthana', 'Hibiscus', 'Jatropha', 'Marigold', 'Rose', 'champaka', 'chitrak', 'honeysuckle', 'indian mallow', 'malabar melastome', 'shankupushpam', 'spider lily', 'sunflower']

# Define class-specific information
class_info = {
    'Common Lanthana': {
        'Description': 'Common Lantana is a flowering plant in the verbena family, native to the American tropics.',
        'Appearance': 'The plant produces small, multicolored flowers, often in shades of red, orange, yellow, pink, and white.',
        'Uses': 'It is widely used as an ornamental plant in gardens and landscapes.',
        'Care': 'It prefers full sun and well-drained soil.'
    },
    'Hibiscus': {
        'Description': 'Hibiscus is a genus of flowering plants in the mallow family, Malvaceae.',
        'Appearance': 'These plants produce large, showy flowers, often in shades of red, pink, orange, yellow, and white.',
        'Uses': 'Hibiscus flowers are used in teas, beverages, and as ornamental plants.',
        'Care': 'They thrive in warm, tropical climates and require plenty of sunlight and water.'
    },
    'Jatropha': {
        'Description': 'Jatropha is a genus of flowering plants in the spurge family, Euphorbiaceae.',
        'Appearance': 'The plant produces clusters of small flowers, typically red or pink.',
        'Uses': 'Jatropha is often grown for its seeds, which can be processed into biofuel.',
        'Care': 'It prefers warm, arid climates and well-drained soil.'
    },
    'Marigold': {
        'Description': 'Marigolds are flowering plants in the daisy family, Asteraceae.',
        'Appearance': 'They produce bright, colorful flowers, typically in shades of orange, yellow, and red.',
        'Uses': 'Marigolds are commonly used as ornamental plants and in companion planting for pest control.',
        'Care': 'They thrive in full sun and well-drained soil.'
    },
    'Rose': {
        'Description': 'Roses are woody perennial flowering plants in the genus Rosa.',
        'Appearance': 'They produce fragrant flowers in a wide range of colors, including red, pink, white, yellow, and orange.',
        'Uses': 'Roses are popular as ornamental plants, in perfumes, and for their culinary uses.',
        'Care': 'They require full sun, well-drained soil, and regular pruning.'
    },
    'champaka': {
        'Description': 'Champaka is a species of flowering plant in the magnolia family, Magnoliaceae.',
        'Appearance': 'The plant produces highly fragrant yellow or white flowers.',
        'Uses': 'The flowers are used in perfumes and traditional ceremonies.',
        'Care': 'It thrives in warm, humid climates and well-drained soil.'
    },
    'chitrak': {
        'Description': 'Chitrak, also known as white leadwort, is a species of flowering plant in the plumbago family, Plumbaginaceae.',
        'Appearance': 'The plant produces small, white or blue flowers.',
        'Uses': 'It is used in traditional medicine for various ailments.',
        'Care': 'It prefers warm climates and well-drained soil.'
    },
    'honeysuckle': {
        'Description': 'Honeysuckle is a genus of arching shrubs or twining vines in the family Caprifoliaceae.',
        'Appearance': 'The plant produces sweetly scented, tubular flowers, often in shades of white, yellow, pink, or red.',
        'Uses': 'Honeysuckle flowers are used in traditional medicine and as ornamental plants.',
        'Care': 'They thrive in full sun to partial shade and well-drained soil.'
    },
    'indian mallow': {
        'Description': 'Indian Mallow is a species of flowering plant in the mallow family, Malvaceae.',
        'Appearance': 'The plant produces yellow, bell-shaped flowers.',
        'Uses': 'It is used in traditional medicine for various ailments.',
        'Care': 'It prefers warm climates and well-drained soil.'
    },
    'malabar melastome': {
        'Description': 'Malabar Melastome is a species of flowering plant in the family Melastomataceae.',
        'Appearance': 'The plant produces purple or pink flowers.',
        'Uses': 'It is used in traditional medicine and as an ornamental plant.',
        'Care': 'It thrives in tropical climates and well-drained soil.'
    },
    'shankupushpam': {
        'Description': 'Shankupushpam, also known as butterfly pea, is a species of flowering plant in the pea family, Fabaceae.',
        'Appearance': 'The plant produces striking blue, white, or purple flowers.',
        'Uses': 'It is used in traditional medicine and as a natural food coloring.',
        'Care': 'It prefers warm climates and well-drained soil.'
    },
    'spider lily': {
        'Description': 'Spider Lilies are flowering plants in the family Amaryllidaceae.',
        'Appearance': 'The plant produces unique, spider-like flowers, often white or yellow.',
        'Uses': 'They are grown as ornamental plants in gardens.',
        'Care': 'They prefer full sun to partial shade and well-drained soil.'
    },
    'sunflower': {
        'Description': 'Sunflowers are flowering plants in the daisy family, Asteraceae.',
        'Appearance': 'They produce large, bright yellow flowers with a central disk.',
        'Uses': 'Sunflowers are grown for their seeds, oil, and as ornamental plants.',
        'Care': 'They thrive in full sun and well-drained soil.'
    }
}

# Define FastAPI app
app = FastAPI()


# Prediction function
def predict(image: Image.Image):
    img = image.resize((640, 640))  # Resize image to match model input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# Define FastAPI route
@app.post("/predict/", response_model=List[dict])
async def predict_endpoint(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            predicted_class, confidence = predict(image)
            result = {"filename": file.filename, "predicted_class": predicted_class, "confidence": confidence}
            if predicted_class in class_info:
                result["class_info"] = class_info[predicted_class]
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return results

