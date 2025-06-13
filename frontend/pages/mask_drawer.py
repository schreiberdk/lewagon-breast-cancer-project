import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Helper: Convert PIL image to base64-encoded data URI ---
def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

# --- Load files ---
st.title("🖌️ Draw Your Own Cancer Mask")

# --- List available image files ---
image_files = sorted(glob.glob("frontend/images/*.png"))  # or .jpg, etc.

if not image_files:
    st.error("No images found in 'images/' folder.")
    st.stop()

# Extract just filenames for display
image_names = [os.path.basename(path) for path in image_files]

# Dropdown to select image
selected_image_name = st.selectbox("Select an image to annotate", image_names)

# Build paths for selected image and corresponding mask
selected_image_path = os.path.join("frontend/images", selected_image_name)
selected_mask_path = os.path.join("frontend/masks", selected_image_name)  # assumes mask has same filename in 'masks/'

# Check files exist
if not os.path.exists(selected_image_path) or not os.path.exists(selected_mask_path):
    st.error(f"Image or mask not found for selection: {selected_image_name}")
    st.stop()

# Load selected images
image = Image.open(selected_image_path).resize((512, 512))
mask_gt = Image.open(selected_mask_path).convert("L").resize((512, 512))

# Convert image to data URI
bg_image_url = image_to_data_url(image)

# --- Drawing canvas ---
st.subheader("1️⃣ Draw your cancer mask on the mammogram:")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.4)",
    stroke_width=5,
    stroke_color="#FF0000",
    background_image=image,  # Pass PIL Image directly here
    update_streamlit=True,
    height=512,
    width=512,
    drawing_mode="freedraw",
    key=f"canvas_{selected_image_name}"
)


def overlay_mask_on_image(image_pil, mask_np, mask_color=(255, 0, 0, 100)):
    """
    Overlay a binary mask on top of a PIL image with transparency.

    Args:
        image_pil (PIL.Image): Original RGB image.
        mask_np (np.ndarray): 2D binary mask (0/1 or 0/255).
        mask_color (tuple): RGBA color for mask overlay (default: semi-transparent red).

    Returns:
        PIL.Image: Image with mask overlay.
    """
    # Make sure mask is binary 0 or 255 for alpha
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")

    # Create an RGBA version of the original image
    image_rgba = image_pil.convert("RGBA")

    # Create a color image the same size as mask
    color_mask = Image.new("RGBA", image_pil.size, mask_color)

    # Put mask as alpha channel on the color mask
    color_mask.putalpha(mask_img)

    # Composite the mask onto the original image
    combined = Image.alpha_composite(image_rgba, color_mask)

    return combined



# --- Process drawn mask ---
if st.button("Submit Mask"):
    if canvas_result.image_data is not None:
        alpha_channel = canvas_result.image_data[:, :, 3]
        if np.any(alpha_channel > 0):
            # Proceed with processing user mask
            user_mask = canvas_result.image_data[:, :, 3] > 0
            user_mask = user_mask.astype(np.uint8)

            # Prepare ground truth mask binary array
            mask_gt_np = np.array(mask_gt)
            mask_gt_bin = (mask_gt_np > 127).astype(np.uint8)

            # Compute IOU
            intersection = np.logical_and(user_mask, mask_gt_bin).sum()
            union = np.logical_or(user_mask, mask_gt_bin).sum()
            iou = intersection / union if union != 0 else 0.0

            st.subheader(f"2️⃣ IOU Score: **{iou:.2f}**")

            # Overlay masks on mammogram
            user_overlay = overlay_mask_on_image(image, user_mask)
            gt_overlay = overlay_mask_on_image(image, mask_gt_bin)

            col1, col2 = st.columns(2)
            with col1:
                st.image(user_overlay, caption="🖌️ Your Mask Overlay", use_column_width=True)
            with col2:
                st.image(gt_overlay, caption="📌 Ground Truth Mask Overlay", use_column_width=True)
    else:
        st.error("Please draw a mask before submitting.")
