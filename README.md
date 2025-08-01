# retinopathy-detection-streamlit
Web app for diabetic retinopathy classification and lesion detection using EfficientNetB4 and YOLOv8, deployed with Streamlit.

This application allows classification of the diabetic retinopathy severity from fundus images and detects lesions using a YOLOv8 model.

# What does the app do?
  - Classifies the degree of eye damage (5 levels of retinopathy) using an _**EfficientNet**_-based classification model.
  - Detects specific lesions (exudates, hemorrhages, optic disc, etc.) using _**YOLOv8**_.
  - Visually displays predictions over the uploaded image.
  - Provides understandable interpretations for non-technical users.

## How to try it?

### Requirements if you want to run it locally  
Install dependencies with:

```bash
pip install -r requirements.txt
```

After that, open your terminal:
- On **Windows**, open **CMD** or **PowerShell**
- On **Linux**, open your **Terminal**

Then run:

```bash
streamlit run app.py
```


