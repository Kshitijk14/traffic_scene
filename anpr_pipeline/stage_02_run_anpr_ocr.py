import os, cv2, csv, math, paddle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from paddleocr import PaddleOCR
import threading
import time
from typing import List, Tuple, Optional


class ANPRPipeline:
    def __init__(self, model_path: str, output_dir: str, csv_path: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.csv_path = Path(csv_path)
        
        # Load YOLO model
        self.yolo_model = YOLO(model_path)
        
        # Thread-local storage for OCR instances to avoid conflicts
        self._local = threading.local()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'no_detections': 0
        }
    
    def get_ocr_instance(self):
        """Get thread-local OCR instance to avoid conflicts"""
        if not hasattr(self._local, 'ocr'):
            use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            self._local.ocr = PaddleOCR(
                use_textline_orientation=True, 
                lang='en',
                use_gpu=use_gpu,
                show_log=False  # Suppress logs for cleaner output
            )
        return self._local.ocr
    
    def validate_crop(self, cropped_img: np.ndarray, min_area: int = 100) -> bool:
        """Validate if cropped image is suitable for OCR"""
        if cropped_img is None or cropped_img.size == 0:
            return False
        
        h, w = cropped_img.shape[:2]
        if h < 10 or w < 20 or (h * w) < min_area:
            return False
        
        # Check if image is not completely black or white
        mean_val = np.mean(cropped_img)
        if mean_val < 5 or mean_val > 250:
            return False
        
        return True
    
    def preprocess_plate(self, cropped_img: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced preprocessing with validation"""
        try:
            if not self.validate_crop(cropped_img):
                return None
            
            # Convert to grayscale
            if len(cropped_img.shape) == 3:
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cropped_img.copy()
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize for better OCR (maintain aspect ratio)
            h, w = cleaned.shape
            if h < 32:  # Minimum height for good OCR
                scale_factor = 32 / h
                new_w = int(w * scale_factor)
                cleaned = cv2.resize(cleaned, (new_w, 32), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to BGR for PaddleOCR
            processed = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            return processed
            
        except Exception as e:
            print(f"[DEBUG] Preprocessing failed: {e}")
            return None
    
    def safe_ocr_predict(self, processed_img: np.ndarray, max_retries: int = 2) -> str:
        """Safely perform OCR with retries and error handling"""
        if processed_img is None:
            return ""
        
        ocr = self.get_ocr_instance()
        
        for attempt in range(max_retries + 1):
            try:
                # Convert BGR to RGB for PaddleOCR
                rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
                # Perform OCR
                result = ocr.ocr(rgb_img, cls=False)
                
                # Extract text safely
                if result and isinstance(result, list) and len(result) > 0:
                    if result[0] and isinstance(result[0], list) and len(result[0]) > 0:
                        # Get the text with highest confidence
                        texts_with_conf = [(item[1][0], item[1][1]) for item in result[0] if item[1][1] > 0.5]
                        if texts_with_conf:
                            # Return text with highest confidence
                            best_text = max(texts_with_conf, key=lambda x: x[1])[0]
                            return best_text.strip()
                
                return ""
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"[DEBUG] OCR attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    print(f"[DEBUG] OCR failed after {max_retries + 1} attempts: {e}")
                    return ""
        
        return ""
    
    def process_single_detection(self, img: np.ndarray, box: np.ndarray, 
                               img_path: str, detection_idx: int) -> Tuple[str, str]:
        """Process a single detection with comprehensive error handling"""
        try:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Validate bounding box
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return Path(img_path).name, ""
            
            # Extract crop
            cropped = img[y1:y2, x1:x2]
            
            # Preprocess
            processed = self.preprocess_plate(cropped)
            if processed is None:
                return Path(img_path).name, ""
            
            # Save debug image
            debug_path = self.output_dir / f"debug_{Path(img_path).stem}_{detection_idx}.jpg"
            cv2.imwrite(str(debug_path), processed)
            
            # Perform OCR
            plate_text = self.safe_ocr_predict(processed)
            
            if plate_text:
                self.stats['successful_ocr'] += 1
                print(f"[SUCCESS] {Path(img_path).name}: '{plate_text}'")
            else:
                self.stats['failed_ocr'] += 1
                print(f"[WARN] No text detected in {Path(img_path).name}")
            
            return Path(img_path).name, plate_text
            
        except Exception as e:
            self.stats['failed_ocr'] += 1
            print(f"[ERROR] Detection processing failed for {img_path}: {e}")
            return Path(img_path).name, ""
    
    def process_image_batch(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        """Process a batch of images with YOLO detection + OCR"""
        batch_results = []
        
        try:
            # Load images
            images = []
            valid_paths = []
            
            for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    valid_paths.append(path)
                else:
                    print(f"[WARN] Could not load image: {path}")
                    batch_results.append((Path(path).name, ""))
            
            if not images:
                return batch_results
            
            # YOLO batch detection
            yolo_results = self.yolo_model.predict(
                source=images, 
                conf=0.5, 
                save=False, 
                verbose=False
            )
            
            # Process each image's detections
            for img, yolo_result, img_path in zip(images, yolo_results, valid_paths):
                self.stats['total_processed'] += 1
                
                if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                    self.stats['no_detections'] += 1
                    batch_results.append((Path(img_path).name, ""))
                    print(f"[INFO] No plates detected in {Path(img_path).name}")
                    continue
                
                # Process all detections for this image
                boxes = yolo_result.boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    result = self.process_single_detection(img, box, img_path, i)
                    batch_results.append(result)
                    
                    # Draw results on image
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, result[1], (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Save annotated image
                output_path = self.output_dir / Path(img_path).name
                cv2.imwrite(str(output_path), img)
        
        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            # Return empty results for failed batch
            for path in image_paths:
                batch_results.append((Path(path).name, ""))
        
        return batch_results
    
    def process_directory(self, img_dir: str, batch_size: int = 4, max_workers: int = 2):
        """Process all images in directory with optimized batching"""
        img_dir = Path(img_dir)
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Find all images
        img_files = [
            str(img_dir / f) for f in os.listdir(img_dir) 
            if f.lower().endswith(img_extensions)
        ]
        
        if not img_files:
            print(f"[WARN] No images found in {img_dir}")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        total_images = len(img_files)
        num_batches = math.ceil(total_images / batch_size)
        all_results = []
        
        print(f"[INFO] Processing {total_images} images in {num_batches} batches...")
        print(f"[INFO] Batch size: {batch_size}, Max workers: {max_workers}")
        
        # Create batches
        batches = [
            img_files[i:i + batch_size] 
            for i in range(0, total_images, batch_size)
        ]
        
        # Process batches with limited threading to avoid OCR conflicts
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_image_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            with tqdm(total=num_batches, desc="Processing Batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.set_postfix({
                            'Success': self.stats['successful_ocr'],
                            'Failed': self.stats['failed_ocr'],
                            'No Detection': self.stats['no_detections']
                        })
                    except Exception as e:
                        print(f"[ERROR] Batch {batch_idx} failed: {e}")
                    finally:
                        pbar.update(1)
        
        # Save results
        self.save_results(all_results)
        self.print_statistics()
    
    def save_results(self, results: List[Tuple[str, str]]):
        """Save results to CSV"""
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "detected_plate"])
                writer.writerows(results)
            
            print(f"[INFO] Results saved to {self.csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total Images Processed: {self.stats['total_processed']}")
        print(f"Successful OCR: {self.stats['successful_ocr']}")
        print(f"Failed OCR: {self.stats['failed_ocr']}")
        print(f"No Detections: {self.stats['no_detections']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful_ocr'] / self.stats['total_processed']) * 100
            print(f"Success Rate: {success_rate:.2f}%")
        print("="*50)


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "../runs/detect/anpr_v2/weights/best.pt"
    IMG_DIR = "../dataset/anpr/processed/images/test/"
    OUTPUT_DIR = "../results/anpr/paddleOCR/test/"
    CSV_PATH = "../results/anpr/paddleOCR/test_logs/results.csv"
    
    # Initialize pipeline
    pipeline = ANPRPipeline(MODEL_PATH, OUTPUT_DIR, CSV_PATH)
    
    # Process images
    # Reduced batch_size and max_workers to avoid threading conflicts
    pipeline.process_directory(
        img_dir=IMG_DIR,
        batch_size=4,  # Smaller batches
        max_workers=2  # Limited threading
    )