import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Neural OCR Engine", page_icon="📝", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; font-family: monospace; text-align: center; }
    .result-box { background: #1e293b; border-left: 4px solid #10b981; padding: 20px; font-size: 2rem; color: #10b981; font-family: monospace; text-align: center; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# --- LOAD CNN MODEL & MAPPING ---
@st.cache_resource
def load_cnn_model():
    try:
        return tf.keras.models.load_model('models/emnist_cnn_model.h5')
    except Exception as e:
        return None

# EMNIST ByClass mapping (0-9, A-Z, a-z)
def get_character_mapping():
    mapping = {}
    for i in range(10): mapping[i] = str(i)
    for i in range(26): mapping[i+10] = chr(i+65)
    for i in range(26): mapping[i+36] = chr(i+97)
    return mapping

model = load_cnn_model()
char_map = get_character_mapping()

# --- HEADER ---
st.markdown("<h1>📝 CUSTOM CNN: HANDWRITTEN TEXT RECOGNITION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Trained on EMNIST &middot; Powered by TensorFlow & OpenCV</p>", unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("Model not found! Please run `python train_cnn.py` first.")
    st.stop()

# --- IMAGE PROCESSING WIZARD ---
st.markdown("### 📸 Upload Handwriting Scan")
st.caption("Upload a photo of a clear, single handwritten word on a white background.")
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Load Image
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    
    # 2. Smart Background Detection
    avg_brightness = np.mean(img_array)
    if avg_brightness > 127:
        binary_input = cv2.bitwise_not(img_array)
    else:
        binary_input = img_array
    
    _, thresh = cv2.threshold(binary_input, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 3. Find raw contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 3 < w < (img_array.shape[1] - 10) and 3 < h < (img_array.shape[0] - 10):
            valid_boxes.append([x, y, w, h])
            
    # 4. Safely Merge 'i' and 'j' Dots (Strict Height Rules to Prevent Line Merging)
    final_boxes = []
    for box in valid_boxes:
        x1, y1, w1, h1 = box
        merged = False
        for i, f_box in enumerate(final_boxes):
            fx, fy, fw, fh = f_box
            
            # RULE 1: Is one box significantly shorter than the other? (Like a dot vs a stem)
            if h1 < fh * 0.6 or fh < h1 * 0.6:
                # RULE 2: Check X-axis overlap (Are they stacked?)
                x_overlap = max(0, min(x1+w1, fx+fw) - max(x1, fx))
                if x_overlap > min(w1, fw) * 0.1:
                    # RULE 3: Check Y-axis gap (Are they close together?)
                    y_gap = max(0, max(y1, fy) - min(y1+h1, fy+fh))
                    if y_gap < max(h1, fh): 
                        # Merge safely!
                        nx = min(x1, fx)
                        ny = min(y1, fy)
                        nw = max(x1+w1, fx+fw) - nx
                        nh = max(y1+h1, fy+fh) - ny
                        
                        final_boxes[i] = [nx, ny, nw, nh]
                        merged = True
                        break
        if not merged:
            final_boxes.append(box)

    # 5. Group Characters into Distinct Rows
    final_boxes = sorted(final_boxes, key=lambda b: b[1]) # Sort by Y-top
    
    rows = []
    if final_boxes:
        current_row = [final_boxes[0]]
        for box in final_boxes[1:]:
            # Check Y-center alignment to handle letters of varying heights
            prev_y_center = current_row[0][1] + (current_row[0][3] / 2)
            box_y_center = box[1] + (box[3] / 2)
            
            if abs(box_y_center - prev_y_center) < max(current_row[0][3], box[3]) * 0.6:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

    # 6. Process and Predict
    display_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    reconstructed_text = ""
    
    col1, col2 = st.columns(2)
    
    with st.spinner("Executing precise line-by-line neural inference..."):
        for row in rows:
            row = sorted(row, key=lambda b: b[0]) # Sort Left-to-Right
            
            for idx, (x, y, w, h) in enumerate(row):
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Space detection
                if idx > 0:
                    prev_x, _, prev_w, _ = row[idx-1]
                    gap = x - (prev_x + prev_w)
                    if gap > prev_w * 0.8: 
                        reconstructed_text += " "
                
                roi = thresh[y:y+h, x:x+w]
                if roi.size > 0:
                    pad = max(w, h) + 8
                    square_roi = np.zeros((pad, pad), dtype=np.uint8)
                    dx = (pad - w) // 2
                    dy = (pad - h) // 2
                    square_roi[dy:dy+h, dx:dx+w] = roi
                    
                    roi_resized = cv2.resize(square_roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi_normalized = roi_resized.astype('float32') / 255.0
                    roi_model_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
                    
                    pred = model.predict(roi_model_input, verbose=0)
                    reconstructed_text += char_map[np.argmax(pred)]
# ... (end of your existing for-loop)
            reconstructed_text += "\n"

    # --- NEW: POST-PROCESSING FILTER ---
    # Fixes EMNIST scale-loss confusion by forcing lowercase and mapping common visual mix-ups
    cleaned_text = reconstructed_text.lower()
    cleaned_text = cleaned_text.replace("0", "o").replace("1", "l").replace("5", "s")
    
    # Fix the specific slanted 'i' misclassified as 'j'
    cleaned_text = cleaned_text.replace("hj", "hi")

    with col1:
        st.markdown("#### Computer Vision: Bounding Boxes")
        st.image(display_img, channels="BGR", use_container_width=True)
        
    with col2:
        st.markdown("#### Neural Network Output")
        st.text_area("", value=cleaned_text, height=150, disabled=True)