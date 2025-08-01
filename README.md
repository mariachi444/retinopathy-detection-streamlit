# retinopathy-detection-streamlit
Web app for diabetic retinopathy classification and lesion detection using EfficientNetB4 and YOLOv8, deployed with Streamlit.



## Overview

This project was developed as part of an assignment for my Biomedical Engineering degree.  
It uses deep learning to detect diabetic retinopathy by combining image classification and lesion detection in a simple web interface.

# What does the app do?
  - Classifies the degree of eye damage (5 levels of retinopathy) using an _**EfficientNet**_-based classification model.
  - Detects specific lesions (exudates, hemorrhages, optic disc, etc.) using _**YOLOv8**_.
  - Provides understandable interpretations for non-technical users.

## How to try it?

Clone this repository with:
```bash
git clone https://github.com/mariachi444/retinopathy-detection-streamlit.git
```

Navigate to the created folder using:
```bash
cd retinopathy-detection-streamlit
```

**Important:** It is recommended to create a virtual environment before installing the dependencies.  
This helps to keep your Python packages organized and avoids conflicts between projects.

You can create and activate a virtual environment with:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```


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



## TODOs
- [ ] Improve EfficientNet classification accuracy
- [ ] Improve YOLOv8 lesion detection accuracy  
- [ ] Deploy to Streamlit Cloud

## Limitations

- Only supports single-image input; batch processing is not available.
- The model was trained only on the APTOS 2019 dataset and **may not** generalize correctly to other image sources.
- This app currently doesn’t support a multilingual interface, and there are no plans to add one.





## License

Feel free to use, modify, or share any part of this project as you wish — no permission or credit is required.

