# frontend/gradio_app.py
import gradio as gr
from PIL import Image
from fastai.vision.all import PILImage

from ml.inference import learn
from ml.config import TARGET_COLS


def predict(img: Image.Image):
    """
    Gradio callback: takes a Pillow image, converts to fastai PILImage,
    returns a dict of class -> probability.
    """
    # Convert Pillow image to fastai PILImage
    img_fai = PILImage.create(img)

    pred_class, pred_idx, probs = learn.predict(img_fai)

    class_names = TARGET_COLS
    prob_dict = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    return prob_dict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload leaf image"),
    outputs=gr.Label(num_top_classes=4, label="Class probabilities"),
    title="Plant Pathology Classifier",
    description="Upload a leaf image to see predicted disease probabilities.",
)


if __name__ == "__main__":
    demo.launch()
