import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---  CONFIGURACIÓN GENERAL --- #
st.set_page_config(
    page_title="Clasificación de Retinopatía + Detección de Lesiones",
    page_icon="👁️",
    layout="centered"
)

IMG_SIZE = 380
CLASES = ['No DR(Clase 0)', 'Mild(Clase 1)', 'Moderate(Clase 2)', 'Severe(Clase 3)', 'Proliferative DR(Clase 4)']
EMOJIS = ["🟢", "🟡", "🟠", "🔴", "⚫"]

# --- 🔹 FUNCIÓN DE PREPROCESAMIENTO--- #
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if mask.any():
            img_stack = [
                img[:, :, c][np.ix_(mask.any(1), mask.any(0))]
                for c in range(3)
            ]
            img = np.stack(img_stack, axis=-1)
        return img
    return img

def enhance_image(img):
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    return img

# --- CARGA DE MODELOS --- #
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model("retina_model_base.keras")

@st.cache_resource
def load_detection_model():
    # Ajusta el nombre de tu .pt si lo guardaste con otra etiqueta
    return YOLO("yolov8m_idrid.pt")

clf_model = load_classification_model()
det_model = load_detection_model()

# --- UI DE BIENVENIDA --- #
st.markdown("<h1 style='text-align: center;'>👁️ Diagnóstico de Retinopatía + Detección de Lesiones</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube una imagen de fondo de ojo para clasificar el grado de retinopatía y detectar lesiones con YOLOv8.</p>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### 📋 Instrucciones")
st.markdown("""
- El sistema ejecutará **dos** tareas:
  1. Clasificación de etapa de retinopatía (con un modelo EfficientNet + Keras).  
  2. Detección de lesiones (con unmodelo YOLOv8 + PyTorch).  
- Sube imágenes claras de fondo de ojo (JPG o PNG).
""")

# --- 🔹 SUBIDA Y PROCESAMIENTO DE IMAGEN --- #
uploaded_file = st.file_uploader("📁 Sube tu imagen aquí", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Leer imagen y mostrar original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen cargada", use_column_width=True)

    # 2. Procesamiento y clasificación
    with st.spinner("🔎 Clasificando retinopatía..."):
        img_np = np.array(image)
        img_np_clf = enhance_image(img_np.copy())
        img_np_clf = preprocess_input(img_np_clf.astype(np.float32))
        img_np_clf = np.expand_dims(img_np_clf, axis=0)

        pred = clf_model.predict(img_np_clf, verbose=0)[0]
        cls_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
    
    emoji = EMOJIS[cls_idx]
    label = CLASES[cls_idx]
    st.success(f"{emoji} **Clasificación:** {label}  —  Confianza: {confidence:.1%}")

    st.bar_chart(pred, height=200)

    st.markdown("---")

    with st.spinner("🩺 Detectando lesiones con YOLOv8..."):
        results = det_model(np.array(image))
        result = results[0]

        # mapeo de etiquetas a nombres amigables
        friendly_labels = {
            "HE": "Hemorragia",
            "EX": "Exudado duro",
            "MA": "Microaneurisma",
            "SE": "Exudado blando",
            "OD": "Disco óptico"
        }

        lesion_counts = {}
        for box in result.boxes:
            cls = result.names[int(box.cls[0])]
            lesion_name = friendly_labels.get(cls, cls)
            lesion_counts[lesion_name] = lesion_counts.get(lesion_name, 0) + 1

        annotated = result.plot()
        annotated_rgb = annotated[..., ::-1]

    # Mostrar imagen con detecciones
    st.image(annotated_rgb, caption="🖼️ Detección de lesiones (YOLOv8)", use_column_width=True)

    # resumen de lesiones
    total_lesiones = sum(lesion_counts.values())
    evaluacion = "🟢 Retina en buen estado" if total_lesiones == 0 else (
                 "🟡 Algunas lesiones detectadas" if total_lesiones <= 5 else
                 "🔴 Daño severo detectado")

    st.markdown(f"**🧾 Resumen de lesiones detectadas:**")
    for lesion, count in lesion_counts.items():
        st.markdown(f"- **{lesion}**: {count}")

    st.markdown(f"**🔍 Evaluación general:** {evaluacion}")

    if lesion_counts:
        # Mostrar gráfico de barras de lesiones
        st.markdown("#### 📊 Distribución de lesiones")
        fig, ax = plt.subplots()
        ax.bar(lesion_counts.keys(), lesion_counts.values(), color='teal')
        ax.set_ylabel("Cantidad")
        ax.set_title("Lesiones detectadas por tipo")
        plt.xticks(rotation=45)
        st.pyplot(fig)

