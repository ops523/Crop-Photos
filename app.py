import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import pillow_avif
import io
import zipfile
import os

st.set_page_config(
    page_title="AI Person Cropper",
    page_icon="✂️",
    layout="wide"
)

# ----------------------------
# Pose Model
# ----------------------------
@st.cache_resource
def load_pose_model():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

pose_model = load_pose_model()

# ----------------------------
# Aspect Ratio Helper
# ----------------------------
def apply_aspect_ratio(image, bbox, aspect_ratio):
    x1, y1, x2, y2 = bbox

    crop_w = x2 - x1
    crop_h = y2 - y1

    if aspect_ratio == "Original":
        return bbox

    ratios = {
        "1:1": 1 / 1,
        "4:5": 4 / 5,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "3:2": 3 / 2
    }

    target_ratio = ratios[aspect_ratio]
    current_ratio = crop_w / crop_h

    if current_ratio > target_ratio:
        new_h = int(crop_w / target_ratio)
        diff = new_h - crop_h

        y1 -= diff // 2
        y2 += diff // 2

    else:
        new_w = int(crop_h * target_ratio)
        diff = new_w - crop_w

        x1 -= diff // 2
        x2 += diff // 2

    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return [x1, y1, x2, y2]

# ----------------------------
# Detect Person
# ----------------------------
def get_crop_box(image, crop_mode, buffer_pixels):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose_model.process(rgb)

    if not results.pose_landmarks:
        return None

    h, w = image.shape[:2]

    landmarks = results.pose_landmarks.landmark

    visible_points = []

    for lm in landmarks:
        if lm.visibility > 0.4:
            visible_points.append(
                (
                    int(lm.x * w),
                    int(lm.y * h)
                )
            )

    if len(visible_points) < 5:
        return None

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    x1 = min(xs)
    x2 = max(xs)

    y1 = min(ys)

    if crop_mode == "Waist Up":

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        hip_y = int(
            ((left_hip.y + right_hip.y) / 2) * h
        )

        y2 = hip_y

    else:
        y2 = max(ys)

    x1 -= buffer_pixels
    y1 -= buffer_pixels
    x2 += buffer_pixels
    y2 += buffer_pixels

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return [x1, y1, x2, y2]

# ----------------------------
# Crop Function
# ----------------------------
def crop_person(image, crop_mode, buffer_pixels, aspect_ratio):
    bbox = get_crop_box(
        image,
        crop_mode,
        buffer_pixels
    )

    if bbox is None:
        return None

    bbox = apply_aspect_ratio(
        image,
        bbox,
        aspect_ratio
    )

    x1, y1, x2, y2 = bbox

    crop = image[y1:y2, x1:x2]

    return crop

# ----------------------------
# UI
# ----------------------------
st.title("✂️ AI Person Cropper")

st.write(
    "Upload images and automatically crop around the detected person."
)

uploaded_files = st.file_uploader(
    "Upload Images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tiff",
        "avif"
    ],
    accept_multiple_files=True
)

crop_mode = st.radio(
    "Crop Mode",
    [
        "Full Body",
        "Waist Up"
    ]
)

buffer_pixels = st.number_input(
    "Buffer Pixels",
    min_value=0,
    max_value=1000,
    value=100,
    step=10
)

aspect_ratio = st.selectbox(
    "Aspect Ratio",
    [
        "Original",
        "1:1",
        "4:5",
        "16:9",
        "9:16",
        "3:2"
    ]
)

# ----------------------------
# Process
# ----------------------------
if uploaded_files:

    if st.button("Process Images"):

        zip_buffer = io.BytesIO()

        processed_count = 0

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            progress = st.progress(0)

            for idx, file in enumerate(uploaded_files):

                try:
                    pil_img = Image.open(file).convert("RGB")

                    image = cv2.cvtColor(
                        np.array(pil_img),
                        cv2.COLOR_RGB2BGR
                    )

                    cropped = crop_person(
                        image,
                        crop_mode,
                        buffer_pixels,
                        aspect_ratio
                    )

                    if cropped is None:
                        continue

                    cropped_rgb = cv2.cvtColor(
                        cropped,
                        cv2.COLOR_BGR2RGB
                    )

                    output_pil = Image.fromarray(
                        cropped_rgb
                    )

                    img_bytes = io.BytesIO()

                    output_pil.save(
                        img_bytes,
                        format="PNG"
                    )

                    filename = (
                        os.path.splitext(
                            file.name
                        )[0]
                        + "_cropped.png"
                    )

                    zip_file.writestr(
                        filename,
                        img_bytes.getvalue()
                    )

                    processed_count += 1

                    progress.progress(
                        (idx + 1)
                        / len(uploaded_files)
                    )

                except Exception as e:
                    st.warning(
                        f"Failed: {file.name}"
                    )

        st.success(
            f"{processed_count} image(s) processed."
        )

        st.download_button(
            "📦 Download ZIP",
            zip_buffer.getvalue(),
            file_name="cropped_images.zip",
            mime="application/zip"
        )
