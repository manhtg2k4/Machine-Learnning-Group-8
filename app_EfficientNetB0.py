import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
from PIL import Image
import cv2

# --- 1. Danh sách lớp ---
class_names = ["chicken", "dog", "cat", "cattle", "elephant"]
NUM_CLASSES = len(class_names)
INPUT_SHAPE = (224, 224, 3)

# --- 2. Hàm xây dựng mô hình giống y hệt code mẫu ---
@st.cache_resource
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = True
    # Freeze các layers ngoại trừ 15 layers cuối
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()
st.success("Mô hình EfficientNetB0 đã sẵn sàng!")

# --- 3. Hàm xử lý ảnh upload ---
def preprocess_image(image: Image.Image):
    # Chuyển sang RGB
    image = image.convert('RGB')
    img_array = np.array(image)
    
    # Resize
    img_array = cv2.resize(img_array, (224, 224))
    
    # Tiền xử lý EfficientNet
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 4. Streamlit UI ---
st.title("Animal Classifier - EfficientNetB0")
st.write("Dự đoán nhãn ảnh từ 5 lớp động vật: chicken, dog, cat, cattle, elephant")

uploaded_file = st.file_uploader("Chọn ảnh để dự đoán", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    img_array = preprocess_image(image)
    
    # Dự đoán
    preds = model.predict(img_array)
    
    # Lấy top-3 dự đoán
    top3_idx = preds[0].argsort()[-3:][::-1]
    st.write("### Top-3 dự đoán:")
    for idx in top3_idx:
        st.write(f"{class_names[idx]}: {preds[0][idx]*100:.2f}% confidence")
    
    # Dự đoán chính
    pred_class = class_names[np.argmax(preds)]
    pred_conf = np.max(preds)
    st.success(f"✅ Dự đoán chính: {pred_class} ({pred_conf*100:.2f}% confidence)")
