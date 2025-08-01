import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import io
import pandas as pd


# ---  CONFIGURACIÓN GENERAL --- #
st.set_page_config(
    page_title="Clasificación de Retinopatía + Detección de Lesiones",
    page_icon="👁️", # También puedes usar una ruta a un archivo .ico/png
    layout="wide", # O "centered" (actual)
    initial_sidebar_state="expanded" # O "collapsed", "auto"
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
    # Forzar 3 canales si la imagen quedó en escala de grises
    if len(img.shape) == 2:  # solo altura x ancho
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # canal único
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    return img

# --- 🔹 FUNCIÓN DE VISUALIZACIÓN MEJORADA CON CLUSTERING --- #
# Definiciones de Clases y Colores (deben coincidir con tu entrenamiento en Colab)
CLASS_NAMES = ["MA", "HE", "EX", "SE", "OD"]
# Colores BGR para OpenCV (coinciden con tu Colab)
CLASS_COLORS_BGR = {
    "MA": (0, 0, 255),    # Rojo
    "HE": (0, 255, 0),    # Verde
    "EX": (255, 0, 0),    # Azul
    "SE": (255, 255, 0),  # Cyan
    "OD": (0, 165, 255)   # Naranja (BGR aproximado)
}

def cluster_and_visualize_for_streamlit(image_np, results, eps=50, min_samples=1):
    """
    Agrupa detecciones cercanas y muestra una caja representativa por grupo.
    Esta versión trabaja directamente con arrays de numpy y objetos de resultado de YOLO.
    """
    # Asegurarse de que la imagen esté en BGR para OpenCV
    if image_np.shape[2] == 3 and np.allclose(image_np[0, 0, :], image_np[0, 0, ::-1]):
        # Parece RGB, convertir a BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.shape[2] == 3:
        # Asumir que ya es BGR
        image_bgr = image_np.copy()
    else:
        # Si hay un problema con los canales, convertir desde RGB por seguridad
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image_display = image_bgr.copy() # Imagen para dibujar

    # Verificar si hay detecciones
    if results[0].boxes is None or len(results[0].boxes) == 0:
        # Si no hay detecciones, devolver la imagen original (convertida a RGB para Streamlit)
        return cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    # Obtener coordenadas de las cajas (xyxy), confianza y clases
    # Nota: results[0].boxes.xyxy ya está en formato xyxy
    try:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy() # Formato xyxy
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        print(f"Error al obtener boxes: {e}")
        return cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB) # Devolver imagen original si hay error

    # Calcular centros de las cajas para clustering
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    centers_y = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    centers = np.column_stack((centers_x, centers_y))

    # Aplicar DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_

    # Dibujar resultados
    unique_labels = set(labels)

    for label in unique_labels:
        class_mask = (labels == label)
        cluster_boxes = boxes_xyxy[class_mask]
        cluster_classes = class_ids[class_mask]
        cluster_confs = confidences[class_mask]

        if label == -1:  # Ruido: dibujar cajas individuales
            for i in range(len(cluster_boxes)):
                x1, y1, x2, y2 = map(int, cluster_boxes[i])
                cls_id = int(cluster_classes[i])
                conf = cluster_confs[i]
                
                # Manejar posibles índices de clase fuera de rango
                if cls_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_id]
                else:
                    class_name = f"Class_{cls_id}"
                    
                color_bgr = CLASS_COLORS_BGR.get(class_name, (255, 255, 255)) # Color BGR

                cv2.rectangle(image_display, (x1, y1), (x2, y2), color_bgr, 2)
                # Ajustar posición del texto si se sale de la imagen
                text = f"{class_name} {conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1] + 5
                cv2.putText(image_display, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        else:  # Cluster: dibujar una caja representativa o mostrar conteo
            if len(cluster_boxes) == 1:
                 # Si el cluster tiene solo una detección, dibujarla normalmente
                x1, y1, x2, y2 = map(int, cluster_boxes[0])
                cls_id = int(cluster_classes[0])
                conf = cluster_confs[0]
                
                if cls_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_id]
                else:
                    class_name = f"Class_{cls_id}"
                    
                color_bgr = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))

                cv2.rectangle(image_display, (x1, y1), (x2, y2), color_bgr, 2)
                text = f"{class_name} {conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1] + 5
                cv2.putText(image_display, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            else:
                # Agrupar múltiples detecciones
                x1_min = int(np.min(cluster_boxes[:, 0]))
                y1_min = int(np.min(cluster_boxes[:, 1]))
                x2_max = int(np.max(cluster_boxes[:, 2]))
                y2_max = int(np.max(cluster_boxes[:, 3]))

                # Determinar la clase representativa (ej: la más común)
                unique_cls, counts_cls = np.unique(cluster_classes, return_counts=True)
                # Clase más frecuente en el cluster
                dominant_class_id = unique_cls[np.argmax(counts_cls)]
                
                if dominant_class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[dominant_class_id]
                else:
                    class_name = f"Class_{dominant_class_id}"
                    
                color_bgr = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))

                # Dibujar la caja del cluster
                cv2.rectangle(image_display, (x1_min, y1_min), (x2_max, y2_max), color_bgr, 2)

                # Mostrar el conteo de elementos en el cluster
                count = len(cluster_boxes)
                text = f"{class_name}_x{count}" # Texto como "MA_x5"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                # Asegurar que el texto no se salga de la imagen
                text_x = x1_min
                text_y = y1_min - 5 if y1_min - 5 > text_size[1] else y1_min + text_size[1] + 5
                cv2.putText(image_display, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    # Convertir de vuelta a RGB para mostrar con matplotlib/Streamlit
    image_rgb_final = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    return image_rgb_final

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
    st.image(image, caption="🖼️ Imagen cargada", use_container_width=True)

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
    # Después de obtener cls_idx, label, confidence, emoji
    col1, col2, col3 = st.columns(3)
    col1.metric("Clase Predicha", f"{emoji} {label}")
    col2.metric("Confianza", f"{confidence:.1%}")
    col3.metric("ID Clase", cls_idx)
    # st.bar_chart(pred, height=200) # Puedes mantener esto también

    st.bar_chart(pred, height=200)

# ... (código de clasificación) ...

    st.markdown("---")

    with st.spinner("🩺 Detectando lesiones con YOLOv8..."):
    # Obtener resultados de YOLO
        results = det_model(np.array(image))
    # No necesitamos 'result = results[0]' aquí porque lo pasamos directamente

        # --- Definiciones para lesiones ---
        # Mapeo de etiquetas a nombres amigables
        friendly_labels = {
            "HE": "Hemorragia",
            "EX": "Exudado duro",
            "MA": "Microaneurisma",
            "SE": "Exudado blando",
            "OD": "Disco óptico"
        }
        # Define explícitamente qué clases son consideradas "lesiones patológicas"
        LESION_TYPES = ["Hemorragia", "Exudado duro", "Microaneurisma", "Exudado blando"]
        # Opcional: Define el nombre del disco óptico para fácil referencia
        OPTIC_DISC_NAME = "Disco óptico"

        # --- Conteo de lesiones (excluyendo el disco óptico del conteo principal) ---
        lesion_counts = {}
        optic_disc_count = 0 # Variable para contar discos ópticos por separado

        # Verificar si hay detecciones antes de iterar
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                # Obtener el nombre de la clase del modelo YOLO entrenado
                cls = results[0].names.get(cls_id, f"Class_{cls_id}")
                lesion_name = friendly_labels.get(cls, cls)

                if lesion_name == OPTIC_DISC_NAME:
                    # Contar disco óptico por separado 
                    optic_disc_count += 1
                else:
                    # Solo contar las lesiones patológicas
                    lesion_counts[lesion_name] = lesion_counts.get(lesion_name, 0) + 1

        
        # Slider para ajustar el agrupamiento
        #st.sidebar.header("🔧 Configuración de Detección")
        #eps_value = st.sidebar.slider("Parámetro de agrupamiento (eps)", min_value=10, max_value=150, value=50, step=5, help="Un valor más alto agrupa más detecciones cercanas.")
        # Aplicar visualización mejorada
        annotated_rgb = cluster_and_visualize_for_streamlit(np.array(image), results, eps=50)
        # --- Fin visualización mejorada ---
        # Después de generar annotated_rgb
        st.markdown("---")
        st.subheader("💾 Descargar Resultados")
        # Descargar imagen anotada
        if st.button("Descargar Imagen Anotada"):
            # Convertir el array numpy RGB a PIL Image
            img_pil = Image.fromarray(annotated_rgb)
            # Guardar en un buffer de bytes
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Imagen como PNG",
                data=byte_im,
                file_name="lesion_detection_result.png",
                mime="image/png"
            )


    # Mostrar imagen con detecciones agrupadas
    st.image(annotated_rgb, caption="🖼️ Detección de lesiones (YOLOv8 - Agrupada)", use_container_width=True)

    list_color = {"Microaneurisma": "Rojo", "Hemorragia": "Verde", "Exudado duro": "Azul", "Exudado blando": "Cyan", "Disco Optico": "Naranja"}


    # Crear dataframe con los colores y clases
    df_colors = pd.DataFrame(list_color.items(), columns=["Lesión", "Color"])
    st.dataframe(df_colors, use_container_width=True, hide_index=True)


    # Mostrar conteo de disco óptico si se detectó
    if optic_disc_count > 0:
        st.info(f"ℹ️ **{OPTIC_DISC_NAME} detectado:** {optic_disc_count}")

    # Mostrar resumen de lesiones patológicas (solo si hay)
    if lesion_counts:
        st.markdown(f"**🧾 Resumen de lesiones detectadas (excluyendo {OPTIC_DISC_NAME}):**")
        # Crear un DataFrame para una mejor presentación
        df_lesions = pd.DataFrame(list(lesion_counts.items()), columns=["Tipo de Lesión", "Cantidad"])
        st.dataframe(df_lesions, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron lesiones patológicas.")
    
    # Evaluación general basada SOLO en lesiones patológicas
    total_lesiones = sum(lesion_counts.values()) # <-- Esta es la línea clave que cambia
    if total_lesiones == 0:
        evaluacion = "🟢 Retina en buen estado"
    elif total_lesiones <= 5: # Ajusta este umbral según tu criterio
        evaluacion = "🟡 Algunas lesiones detectadas"
    else:
        evaluacion = "🔴 Existen lesiones graves"

    st.markdown(f"**🔍 Evaluación general (basada en lesiones):** {evaluacion}")

    # Mostrar gráfico de barras de lesiones patológicas
    if lesion_counts: # <-- Solo mostrar si hay lesiones patológicas
        st.markdown("#### 📊 Distribución de lesiones")
        # Opción 1: Con matplotlib
        fig, ax = plt.subplots()
        ax.bar(list(lesion_counts.keys()), list(lesion_counts.values()), color='teal')
        ax.set_ylabel("Cantidad")
        ax.set_title("Lesiones detectadas por tipo")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        