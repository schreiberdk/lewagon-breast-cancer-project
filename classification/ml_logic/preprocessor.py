# Import necessary libraries
import pandas as pd
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import os
from scipy import ndimage

class Preprocessor:
    """
    Preprocessor class for mammogram images.
    This class provides methods to preprocess mammogram images for machine learning tasks.
    It includes functionality for:
    - Detecting breast regions using advanced gradient-based methods
    - Inverting black-on-white images
    - Normalizing pixel values
    - Applying CLAHE contrast enhancement
    - Padding images to square dimensions
    - Resizing images to target dimensions
    - Saving processed images in a structured output directory
    """

    def __init__(self):
        """
        Initialize the preprocessor with optional configuration.
        """

    ## Support functions for preprocessing
    @staticmethod
    def is_black_on_white(image, threshold=0.6):
        """
        Determines if the image is black-on-white (i.e., most pixels are bright).
        threshold: fraction of image area that must be bright for it to be considered white-on-black.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Count pixels above mid-gray (128)
        bright_pixels = np.sum(gray > 127)
        total_pixels = gray.size
        return (bright_pixels / total_pixels) > threshold

    @staticmethod
    def invert_if_black_on_white(image):
        """
        Inverts the image if it is black-on-white.
        """
        if Preprocessor.is_black_on_white(image):
            return cv2.bitwise_not(image)
        return image

    @staticmethod
    def detect_breast_region(image, return_only_cropped_image = False):
        """
        Enhanced gradient-based detection with multiple refinements
        Based on research from mammographic segmentation papers
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Advanced preprocessing (as mentioned in research)
        # Density-weighted contrast enhancement (DWCE)
        # Ensure image is uint8 with proper scaling
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:  # Handle normalized [0,1] float images
                gray = (gray * 255).astype(np.uint8)
            else:  # Handle other data types (e.g., uint16)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Noise reduction with edge preservation
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Step 2: Multi-scale gradient computation
        # Combine different gradient scales for robustness
        gradients = []
        scales = [1, 3, 5]  # Different Sobel kernel sizes

        for ksize in scales:
            grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=ksize)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(gradient_mag)

        # Weighted combination of multi-scale gradients
        combined_gradient = np.zeros_like(gradients[0])
        weights = [0.5, 0.3, 0.2]  # Higher weight for finer scales
        for i, grad in enumerate(gradients):
            combined_gradient += weights[i] * grad

        # Step 3: Adaptive thresholding on gradient
        # Use percentile-based thresholding instead of fixed values
        gradient_norm = ((combined_gradient - combined_gradient.min()) /
                        (combined_gradient.max() - combined_gradient.min()) * 255).astype(np.uint8)

        # Adaptive threshold based on image statistics
        threshold_val = np.percentile(gradient_norm[gradient_norm > 0], 75)
        _, gradient_binary = cv2.threshold(gradient_norm, threshold_val, 255, cv2.THRESH_BINARY)

        # Step 4: Combine with intensity information
        # Otsu on original image
        _, intensity_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine gradient and intensity information
        combined = cv2.bitwise_or(gradient_binary, intensity_binary)

        # Step 5: Advanced morphological operations
        # Use different kernels for different operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        # Close gaps first
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        # Remove small noise
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

        # Step 6: Hole filling and region refinement
        # Fill holes using scipy
        filled = ndimage.binary_fill_holes(combined).astype(np.uint8) * 255 # type: ignore

        # Step 7: Advanced contour filtering
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Multi-criteria filtering
            image_area = gray.shape[0] * gray.shape[1]
            filtered_contours = []

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                # Area filter
                if not (image_area * 0.1 < area < image_area * 0.9):
                    continue

                # Shape filter - breast regions should have reasonable aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if not (0.3 < aspect_ratio < 3.0):
                    continue

                # Compactness filter - avoid very irregular shapes
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)
                    if compactness < 0.1:  # Very irregular shapes
                        continue

                filtered_contours.append(contour)

            if filtered_contours:
                # Select best contour based on position and size
                best_contour = max(filtered_contours, key=cv2.contourArea)

                # Create refined mask
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [best_contour], 255) # type: ignore

                # Post-process mask to smooth boundaries
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_open)

                x, y, w, h = cv2.boundingRect(best_contour)
                if return_only_cropped_image:
                    return gray[y:y+h, x:x+w]  # Return NumPy array
                return mask, (x, y, w, h), best_contour

        # Handle failed detection based on return type
        if return_only_cropped_image:
            return None  # Single None for cropped images
        else:
            return None, None, None  # Tuple for mask/bbox/contour

    @staticmethod
    def detect_breast_region_optimized(image):
        """Fast, robust breast detection using adaptive thresholding + region growing"""
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 1. Median filter for noise removal (faster than bilateral)
        denoised = cv2.medianBlur(image, 5)

        # 2. CLAHE with optimized params from [8]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        enhanced = clahe.apply(denoised)

        # 3. Adaptive thresholding using Otsu's method
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # 4. Largest connected component (fast alternative to contours)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        if num_labels < 2:
            return None  # No breast region found

        # Get largest component excluding background
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_label).astype(np.uint8) * 255

        # 5. Morphological refinement (optimized kernel sizes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    ## Main preprocessing function
    @staticmethod
    def preprocess_dataset(input_dir, output_dir,
                           resize_shape=(224, 224),
                           apply_clahe=True,
                           clahe_clip_limit=2.0,
                           file_exts=('.jpg', '.jpeg', '.png', '.dcm')):
        """
        Full preprocessing pipeline for mammogram images.
        This function processes images in the following way:
        1. Creates output directory with same structure as input
        2. Reads images (supports DICOM and standard formats)
        3. Inverts black-on-white images
        4. Crops breast regions
        5. Normalizes pixel values
        6. Applies CLAHE contrast enhancement if specified
        7. Pads images to square
        8. Resizes to target dimensions
        9. Saves processed images in the output directory

        Args:
            input_dir: Root directory with images
            output_dir: Target directory for processed images
            resize_shape: Final output dimensions (width, height)
            apply_clahe: Whether to apply contrast enhancement
            clahe_clip_limit: CLAHE contrast limit
            file_exts: Supported file extensions
        """

        processed_count = 0
        failed_count = 0

        # 1. Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Walk through directory tree
        for root, dirs, files in os.walk(input_dir):
            # Create mirror directories in output
            relative_path = os.path.relpath(root, input_dir)
            current_out_dir = os.path.join(output_dir, relative_path)
            os.makedirs(current_out_dir, exist_ok=True)

            for file in files:
                if not file.lower().endswith(file_exts):
                    print(f"Skipping unsupported file: {file}")
                    continue

                input_path = os.path.join(root, file)
                output_path = os.path.join(current_out_dir, file)

                try:
                    # 2. Read image (handle DICOM and standard formats)
                    if input_path.lower().endswith('.dcm'):
                        ds = pydicom.dcmread(input_path)
                        img = ds.pixel_array.astype(np.float32)
                        img = (img - img.min()) / (img.max() - img.min()) * 255
                        img = img.astype(np.uint8)
                    else:
                        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        raise ValueError(f"Could not read image from {input_path}")

                    # 3. Invert if black-on-white
                    img = Preprocessor.invert_if_black_on_white(img)

                    # 4. Crop breast region
                    cropped = Preprocessor.detect_breast_region(img, return_only_cropped_image=True)
                    if cropped is None:
                        print(f"Breast region detection failed for {input_path}, using original image.")
                        cropped = img  # Fallback to original if detection fails

                    # 5. Normalize
                    normalized = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX) # type: ignore

                    # 6. Increase contrast with CLAHE (if specified)
                    if apply_clahe:
                        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                                tileGridSize=(8,8))
                        normalized = clahe.apply(normalized)

                    # 7. Pad to square
                    h, w = normalized.shape
                    max_dim = max(h, w)
                    padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
                    y_offset = (max_dim - h) // 2
                    x_offset = (max_dim - w) // 2
                    padded[y_offset:y_offset+h, x_offset:x_offset+w] = normalized

                    # 8. Resize
                    resized = cv2.resize(padded, resize_shape,
                                         interpolation=cv2.INTER_AREA)

                    # 9. Save processed image
                    cv2.imwrite(output_path, resized)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images...")

                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
                    failed_count += 1
                    continue

        print(f"Preprocessing complete!")
        print(f"Successfully processed: {processed_count} images")
        print(f"Failed: {failed_count} images")

    @staticmethod
    def preprocess_image(image,
                         resize_shape=(224, 224),
                         apply_clahe=True,
                         clahe_clip_limit=2.0,
                         file_exts=('.jpg', '.jpeg', '.png', '.dcm'),
                         test_mode=False):
        """
        Full preprocessing pipeline for single mammogram image.
        This function processes images in the following way:
        1. Reads image (supports DICOM and standard formats)
        2. Inverts if black-on-white image
        3. Crops breast regions
        4. Applies CLAHE contrast enhancement if specified
        5. Pads images to square
        6. Resizes to target dimensions
        7. Normalizes pixel values to [0, 1]
        8. Turns image to 3-channel RGB format for ML models
        9. Return processed image as NumPy array

        Args:
            image: Input image (as file path)
            resize_shape: Final output dimensions (width, height)
            apply_clahe: Whether to apply contrast enhancement
            clahe_clip_limit: CLAHE contrast limit
            file_exts: Supported file extensions
            test_mode: If True, returns intermediate steps for debugging
        """

        # Initialize variables
        img = None

        # Case 1: Input is a file path (string)
        if isinstance(image, str):
            if not image.lower().endswith(file_exts):
                print(f"Image format unsupported: {image}")
                return None
            try:
                if image.lower().endswith('.dcm'):
                    import pydicom
                    ds = pydicom.dcmread(image)
                    img = ds.pixel_array.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                else:
                    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not read image from {image}")
            except Exception as e:
                print(f"Error reading {image}: {str(e)}")
                return None

        # Case 2: Input is a NumPy array
        elif isinstance(image, np.ndarray):
            img = image

        # Case 3: File-like object (e.g., BytesIO)
        elif hasattr(image, 'read'):
            # Read bytes, convert to numpy array, decode with OpenCV
            file_bytes = np.frombuffer(image.read(), np.uint16)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Could not decode image from file-like object")
                return None

        else:
            print("Input must be a file path (str), a NumPy array, or a file-like object")
            return None

        if img is None:
            raise ValueError(f"Could not read image from {image}")

        # Convert to uint8 if necessary
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Convert 3-channel → 1-channel

        try:
            # 2. Invert if black-on-white
            img = Preprocessor.invert_if_black_on_white(img)

            # 3. Crop breast region
            mask = Preprocessor.detect_breast_region_optimized(img)
            if mask is not None:
                # Get bounding box from mask
                x, y, w, h = cv2.boundingRect(mask)
                cropped = img[y:y+h, x:x+w]
            else:
                print(f"Breast region detection failed for {img} images, using original image.")
                cropped = img  # Fallback to original if detection fails

            # 4. Increase contrast with CLAHE (if specified)
            if apply_clahe:
                if cropped.dtype != np.uint8:
                    cropped = np.clip(cropped, 0, 255).astype(np.uint8)

                # Check for contrast variation
                if np.ptp(cropped) == 0:  # All pixels identical
                    print("Skipping CLAHE - no contrast variation")
                else:
                    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8))
                    cropped = clahe.apply(cropped)

            # 5. Pad to square
            h, w = cropped.shape
            # Add dimension validation
            if h == 0 or w == 0:
                print(f"Invalid dimensions after cropping: {h}x{w}")
                return None

            # Ensure square padding works for all aspect ratios
            max_dim = max(h, w)
            padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2

            # Handle cases where offset + dimension exceeds bounds
            padded[
                max(0, y_offset):min(y_offset+h, max_dim),
                max(0, x_offset):min(x_offset+w, max_dim)
            ] = cropped[
                max(0, -y_offset):min(h, max_dim-y_offset),
                max(0, -x_offset):min(w, max_dim-x_offset)
            ]

            # 6. Resize
            resized = cv2.resize(padded, resize_shape,
                                 interpolation=cv2.INTER_AREA)

            # 7. Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            # 8. Convert grayscale to 3-channel RGB (for ML models)
            resized_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

            # 8. Return processed image
            if test_mode:
                return resized_rgb, normalized, resized, padded, cropped, mask, img
            return resized_rgb

        except Exception as e:
            print(f"Error processing {image}: {str(e)}")
            return None

# If you want to allow running as a script
if __name__ == "__main__":
    # PLACEHOLDER
    print("Preprocessor module loaded. Use Preprocessor class for preprocessing mammogram images.")
