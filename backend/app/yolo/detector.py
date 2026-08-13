import os
import numpy as np
from typing import List, Dict, Any
from app.config.config import settings


class YOLODamageDetector:
    """
    Multi-Model Ultralytics YOLO Engine:
    1. best.pt -> Road damage detection (pothole, longitudinal_crack, transverse_crack, alligator_crack, missing_asphalt, broken_road)
    2. yolov8n.pt -> Vehicle detection (car, truck, bus, motorcycle, bicycle, person)
    3. numberplate.pt -> Number plate detection (number_plate)

    All models are loaded ONCE during application startup and executed in memory per frame.
    """

    ROAD_DAMAGE_CLASSES = {
        0: "pothole",
        1: "longitudinal_crack",
        2: "transverse_crack",
        3: "alligator_crack",
        4: "missing_asphalt",
        5: "broken_road"
    }

    COCO_VEHICLE_MAP = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self, model_path: str = settings.YOLO_MODEL_PATH):
        self.model_path = model_path
        self.damage_model = None
        self.vehicle_model = None
        self.plate_model = None
        self._load_all_models()

    def _load_all_models(self):
        """
        Load all three YOLO models once during application startup.
        Models are kept in memory to optimize FPS and prevent reload overhead.
        1. best.pt -> Road damage detection
        2. yolov8n.pt -> Vehicle detection
        3. helmet_numberplate.pt -> Helmet & Number Plate detection
        """
        try:
            from ultralytics import YOLO

            # 1. Road Damage Model (best.pt)
            best_file = self.model_path if os.path.exists(self.model_path) else "best.pt"
            if os.path.exists(best_file):
                try:
                    self.damage_model = YOLO(best_file)
                    print(f"Loaded Road Damage YOLO model from {best_file}")
                except Exception as e:
                    print(f"Notice loading {best_file}: {e}")
                    self.damage_model = None
            else:
                print(f"Custom damage model weights '{best_file}' not found locally. Initializing damage model fallback.")
                self.damage_model = None

            # 2. Vehicle Model (yolov8n.pt)
            try:
                self.vehicle_model = YOLO("yolov8n.pt")
                print("Loaded Vehicle Detection YOLO model (yolov8n.pt)")
            except Exception as ve:
                print(f"Notice loading vehicle model yolov8n.pt: {ve}")
                self.vehicle_model = None

            # 3. Helmet & Number Plate Model (helmet_numberplate.pt)
            hp_file = "helmet_numberplate.pt"
            if not os.path.exists(hp_file):
                hp_file = "numberplate.pt"

            if os.path.exists(hp_file):
                try:
                    self.plate_model = YOLO(hp_file)
                    print(f"Loaded Helmet & Number Plate YOLO model from {hp_file}")
                except Exception as pe:
                    print(f"Notice loading {hp_file}: {pe}")
                    self.plate_model = None
            else:
                print(f"Helmet & Number plate weights '{hp_file}' not found locally. Will use heuristics if needed.")
                self.plate_model = None

        except Exception as e:
            print(f"Warning: Failed to load PyTorch Ultralytics YOLO models: {e}. Running in computer vision heuristic mode.")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = settings.CONFIDENCE_THRESHOLD,
        iou_threshold: float = settings.IOU_THRESHOLD,
        roi_horizon_cutoff: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Run 3-step detection pipeline on incoming frame in memory:
        Step 1: Run best.pt -> Detect road damages
        Step 2: Run yolov8n.pt -> Detect vehicles
        Step 3: Run helmet_numberplate.pt -> Detect helmets and number plates
        Merge all detections into a single JSON response array.
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        horizon_y_limit = height * roi_horizon_cutoff
        min_area_pixels = 100
        max_area_pixels = height * width * 0.85

        merged_detections: List[Dict[str, Any]] = []

        # =====================================================
        # STEP 1: Road Damage Detection (best.pt)
        # =====================================================
        if self.damage_model is not None:
            try:
                dmg_results = self.damage_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                if dmg_results and len(dmg_results) > 0:
                    for box in dmg_results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        w, h = x_max - x_min, y_max - y_min
                        area = w * h

                        if y_min < horizon_y_limit and y_max < horizon_y_limit + 20:
                            continue
                        if area < min_area_pixels or area > max_area_pixels:
                            continue

                        if hasattr(self.damage_model, "names") and self.damage_model.names:
                            category = self.damage_model.names.get(cls_id, self.ROAD_DAMAGE_CLASSES.get(cls_id % 6, "pothole"))
                        else:
                            category = self.ROAD_DAMAGE_CLASSES.get(cls_id % 6, "pothole")

                        category_str = str(category).lower()

                        merged_detections.append({
                            "category": category_str,
                            "confidence": round(conf, 4),
                            "type": "damage",
                            "bbox": {
                                "x_min": round(x_min, 2),
                                "y_min": round(y_min, 2),
                                "x_max": round(x_max, 2),
                                "y_max": round(y_max, 2)
                            },
                            "x_min": round(x_min, 2),
                            "y_min": round(y_min, 2),
                            "x_max": round(x_max, 2),
                            "y_max": round(y_max, 2),
                            "area_pixels": round(area, 2)
                        })
            except Exception as err:
                print(f"Road damage model inference runtime notice: {err}")

        # =====================================================
        # STEP 2: Vehicle Detection (yolov8n.pt)
        # Detect only: car, truck, bus, motorcycle, bicycle, person
        # =====================================================
        if self.vehicle_model is not None:
            try:
                veh_results = self.vehicle_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=[0, 1, 2, 3, 5, 7],
                    verbose=False
                )
                if veh_results and len(veh_results) > 0:
                    for box in veh_results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        w, h = x_max - x_min, y_max - y_min
                        area = w * h

                        category = self.COCO_VEHICLE_MAP.get(cls_id, "car")

                        merged_detections.append({
                            "category": category,
                            "confidence": round(conf, 4),
                            "type": "vehicle",
                            "bbox": {
                                "x_min": round(x_min, 2),
                                "y_min": round(y_min, 2),
                                "x_max": round(x_max, 2),
                                "y_max": round(y_max, 2)
                            },
                            "x_min": round(x_min, 2),
                            "y_min": round(y_min, 2),
                            "x_max": round(x_max, 2),
                            "y_max": round(y_max, 2),
                            "area_pixels": round(area, 2)
                        })
            except Exception as err:
                print(f"Vehicle model inference runtime notice: {err}")

        # =====================================================
        # STEP 3: Helmet & Number Plate Detection (helmet_numberplate.pt)
        # Detect: helmet, number_plate
        # =====================================================
        if self.plate_model is not None:
            try:
                hp_results = self.plate_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                if hp_results and len(hp_results) > 0:
                    for box in hp_results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        w, h = x_max - x_min, y_max - y_min
                        area = w * h

                        category_name = "number_plate"
                        det_type = "plate"

                        if hasattr(self.plate_model, "names") and self.plate_model.names:
                            raw_name = str(self.plate_model.names.get(cls_id, "")).lower()
                            if "helmet" in raw_name:
                                category_name = "helmet"
                                det_type = "helmet"
                            elif "plate" in raw_name or "number" in raw_name:
                                category_name = "number_plate"
                                det_type = "plate"
                            else:
                                if cls_id == 0:
                                    category_name = "helmet"
                                    det_type = "helmet"
                                else:
                                    category_name = "number_plate"
                                    det_type = "plate"
                        else:
                            if cls_id == 0:
                                category_name = "helmet"
                                det_type = "helmet"
                            else:
                                category_name = "number_plate"
                                det_type = "plate"

                        merged_detections.append({
                            "category": category_name,
                            "confidence": round(conf, 4),
                            "type": det_type,
                            "bbox": {
                                "x_min": round(x_min, 2),
                                "y_min": round(y_min, 2),
                                "x_max": round(x_max, 2),
                                "y_max": round(y_max, 2)
                            },
                            "x_min": round(x_min, 2),
                            "y_min": round(y_min, 2),
                            "x_max": round(x_max, 2),
                            "y_max": round(y_max, 2),
                            "area_pixels": round(area, 2)
                        })
            except Exception as err:
                print(f"Helmet & Number plate model inference notice: {err}")

        # Computer vision heuristic fallback if no models or zero results returned
        if not merged_detections and self.damage_model is None:
            return self._heuristic_fallback(frame, conf_threshold)

        return merged_detections

    def _heuristic_fallback(self, frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
        """CV fallback heuristics when PyTorch model weights are uninitialized"""
        import cv2
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_y_start = int(height * 0.3)
        roi = gray[roi_y_start:, :]

        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < (width * height * 0.15):
                x, y, w, h = cv2.boundingRect(cnt)
                y += roi_y_start
                aspect_ratio = float(w) / h if h > 0 else 1.0
                conf = min(0.92, max(conf_threshold + 0.05, area / 5000.0))

                if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                    cat = "longitudinal_crack" if aspect_ratio < 0.3 else "transverse_crack"
                    dtype = "damage"
                elif area > 3000:
                    cat = "pothole"
                    dtype = "damage"
                else:
                    cat = "alligator_crack"
                    dtype = "damage"

                detections.append({
                    "category": cat,
                    "confidence": round(conf, 4),
                    "type": dtype,
                    "bbox": {
                        "x_min": float(x),
                        "y_min": float(y),
                        "x_max": float(x + w),
                        "y_max": float(y + h)
                    },
                    "x_min": float(x),
                    "y_min": float(y),
                    "x_max": float(x + w),
                    "y_max": float(y + h),
                    "area_pixels": float(area)
                })
                if len(detections) >= 5:
                    break

        return detections
