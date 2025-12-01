# Plant Pathology Classifier 🌿

Image classifier for the Kaggle Plant Pathology 2020 dataset.  
Upload a leaf image and get predicted probabilities for four classes:

- `healthy`
- `multiple_diseases`
- `rust`
- `scab`

---

## How It’s Made

**Tech used:** Python, fastai, PyTorch, timm, FastAPI, Uvicorn, Gradio, Google Colab, Conda, Git/GitHub

- Trained an EfficientNet-based image classifier on the Kaggle dataset in Google Colab using fastai + timm.
- Exported the trained model as a fastai `.pkl` file and saved it in a local `models/` folder.
- Built a FastAPI backend with a `/predict` endpoint that accepts an image upload and returns JSON with the predicted label and class probabilities.
- Added a simple Gradio frontend so you can upload a leaf image in the browser and see the probabilities for each class.

---

## Optimizations

- Reuse a single loaded fastai learner for both the FastAPI backend and the Gradio UI so the model only loads once.
- Map numeric class indices (`0, 1, 2, 3`) to readable labels (`healthy`, `multiple_diseases`, `rust`, `scab`) for cleaner outputs.
- Matched the local Python/fastai environment to the Colab training environment so the exported `.pkl` loads reliably.

---

## Lessons Learned

- Learned how to turn a fastai notebook model into a reusable inference module.
- Got hands-on experience with FastAPI for serving a real ML model as an HTTP API.
- Learned how to use Gradio to build a quick, user-friendly demo for an ML model.
- Improved fastai skills and practised using Git/GitHub and Conda to structure an end-to-end ML project.

## Future Checklist

### Done ✅

- [x] Train EfficientNet-based model on Kaggle Plant Pathology in Colab  
- [x] Export model to `.pkl` and load it locally for inference  
- [x] Build FastAPI `/predict` endpoint for image uploads and JSON responses  
- [x] Create Gradio UI for uploading leaf images and viewing class probabilities  
- [x] Map numeric class indices to human-readable labels (`healthy`, `multiple_diseases`, `rust`, `scab`)  
- [x] Organise project into clear modules (`ml/`, `app/`, `frontend/`, `models/`) and push to GitHub  

### Planned 🚀

- [ ] Add a `Dockerfile` and containerise the FastAPI backend  
- [ ] Deploy the Dockerised API to a cloud service (e.g. Render or Railway)  
- [ ] Deploy the Gradio app as a Hugging Face Space for an easy online demo  
- [ ] Retrain / tune the model (better hyperparameters, data augments, or architecture tweaks)  
- [ ] Add a simple evaluation script (metrics + confusion matrix)  
- [ ] Add basic tests (pytest) for the API and inference code and wire them into GitHub Actions CI  
