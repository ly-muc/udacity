import argparse
import json
 
import tensorflow as tf
import numpy as np 
import tensorflow_hub as hub
from PIL import Image

def process_image(image):
    
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 255.
    
    return image.numpy()

def predict(image_path, model, top_k):
    
    image = np.array(Image.open(image_path))
    image = process_image(image)
    
    image = image[np.newaxis, ...]
    
    proba_predicted = model.predict(image)
    
    probs, classes = tf.nn.top_k(proba_predicted, k=top_k)
    
    return probs.numpy().flatten().tolist(), classes.numpy().flatten().tolist()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", type=str, default="label_map.json")

    args = parser.parse_args()
    
    # load class names
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    # load model
    model = tf.keras.models.load_model("trained_model.h5", custom_objects={'KerasLayer': hub.KerasLayer}, compile = False)

    # get prediction
    probs, classes = predict(args.image_path, model, args.top_k)
    classes = [class_names[str(c+1)] for c in classes]

    print(f"The top {args.top_k} predictions for the given image are:")
    
    for p, c in zip(probs, classes):
        print(f"{c}: {p*100}%")
    
if __name__ == "__main__":
    main()
    
    
    