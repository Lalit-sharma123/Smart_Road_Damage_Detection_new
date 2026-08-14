import os
import json
import base64
import cv2
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db, AsyncSessionLocal
from app.models.models import (
    User, Video, Frame, Detection, GPSData, RoadAnalytics,
    UserRole, ProcessingStatus, SeverityLevel, DamageCategory
)
from app.schemas.schemas import ProcessVideoRequest, ProcessVideoResponse
from app.auth.jwt import get_current_user_optional
from app.cv.video_processor import VideoProcessor
from app.yolo.detector import YOLODamageDetector
from app.services.severity_service import SeverityAnalysisService
from app.services.gps_service import GPSExtractionService
from app.services.websocket_manager import ws_broadcaster
from app.api.ws_routes import processing_progress_state
from app.config.config import settings

router = APIRouter(prefix="/process", tags=["Processing Engine"])

detector_instance = YOLODamageDetector()

# WebSocket Manager for Live Processing Stages
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()


@router.websocket("/ws")
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
    await ws_manager.connect(websocket)
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "stage": "Connected",
            "progress": 0,
            "message": f"WebSocket connection active for client: {client_id}",
            "timestamp": asyncio.get_event_loop().time()
        }))
        while True:
            data = await websocket.receive_text()
            # Keep-alive echo
            await websocket.send_text(json.dumps({
                "stage": "Connected",
                "progress": processing_progress_state.get("progress_percent", 0),
                "message": f"Channel active ({client_id})",
                "timestamp": asyncio.get_event_loop().time()
            }))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


async def execute_video_processing_task(
    video_id: str,
    confidence_threshold: float = 0.35,
    frame_skip: int = 5,
    enable_histogram_equalization: bool = True,
    enable_gaussian_blur: bool = True
):
    """
    Asynchronous Background Task:
    Executes OpenCV frame extraction, YOLOv11 multi-model inference, severity index computation,
    database persistence, and real-time live frame broadcast over WebSockets.
    """
    async def send_ws_update(stage: str, progress: int, msg: str):
        msg_dict = {
            "stage": stage,
            "progress": progress,
            "message": msg,
            "timestamp": asyncio.get_event_loop().time()
        }
        await ws_manager.broadcast(msg_dict)
        await ws_broadcaster.broadcast(msg_dict)

    async with AsyncSessionLocal() as db:
        video = None
        try:
            stmt = select(Video).where(Video.id == video_id)
            video = (await db.execute(stmt)).scalar_one_or_none()
            if not video:
                print(f"[Processing Task] Video '{video_id}' not found in database.")
                return

            video.status = ProcessingStatus.PROCESSING
            await db.commit()

            await send_ws_update("Uploading", 15, "FastAPI WS: Ingestion verified. Initializing OpenCV decoding...")

            # Initialize VideoProcessor
            processor = VideoProcessor(video.file_path)
            all_detections_list = []
            frames_processed_count = 0

            await send_ws_update("Extracting Frames", 35, f"FastAPI WS: Slicing frames with frame_skip={frame_skip}...")

            # Extract frames generator with frame skip
            frame_gen = processor.extract_frames_generator(
                frame_skip=frame_skip,
                enable_histogram_eq=enable_histogram_equalization,
                enable_gaussian_blur=enable_gaussian_blur
            )

            await send_ws_update("Running YOLO", 50, "FastAPI WS: Executing YOLOv11 damage detection inference...")

            # Output MP4 video writer
            output_mp4_path = os.path.join(settings.PROCESSED_DIR, f"processed_{video.id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_mp4_path,
                fourcc,
                processor.fps or 30.0,
                (processor.width, processor.height)
            )

            total_expected_frames = processor.total_frames or 100

            for frame_num, timestamp_sec, raw_frame, preprocessed_frame in frame_gen:
                frames_processed_count += 1

                # YOLO detection on preprocessed frame
                raw_detections = detector_instance.detect(
                    preprocessed_frame,
                    conf_threshold=confidence_threshold
                )

                has_damage = len(raw_detections) > 0

                # DB Frame and Detection persistence
                try:
                    db_frame = Frame(
                        video_id=video.id,
                        frame_number=frame_num,
                        timestamp_seconds=timestamp_sec,
                        image_path="",
                        has_damage=has_damage
                    )
                    db.add(db_frame)
                    await db.flush()

                    for det in raw_detections:
                        sev_level, sev_score = SeverityAnalysisService.calculate_detection_severity(
                            det,
                            frame_width=processor.width,
                            frame_height=processor.height,
                            cluster_count=len(raw_detections)
                        )

                        cat_val = det["category"]
                        if hasattr(cat_val, "value"):
                            cat_enum = cat_val
                        else:
                            try:
                                cat_enum = DamageCategory(cat_val)
                            except ValueError:
                                cat_enum = DamageCategory.POTHOLE

                        distance_est = SeverityAnalysisService.estimate_perspective_distance(
                            det,
                            frame_height=processor.height
                        )

                        base_lat = 28.4595 + (frame_num * 0.00008)
                        base_lon = 77.0266 + (frame_num * 0.00009)

                        db_detection = Detection(
                            video_id=video.id,
                            camera_id=None,
                            frame_id=db_frame.id,
                            frame_number=frame_num,
                            timestamp_seconds=timestamp_sec,
                            category=cat_enum,
                            confidence=det["confidence"],
                            x_min=det["x_min"],
                            y_min=det["y_min"],
                            x_max=det["x_max"],
                            y_max=det["y_max"],
                            area_pixels=det["area_pixels"],
                            severity=sev_level,
                            severity_score=sev_score,
                            distance_meters=distance_est,
                            latitude=base_lat,
                            longitude=base_lon
                        )
                        db.add(db_detection)

                        det_record = {
                            "category": cat_enum.value,
                            "confidence": det["confidence"],
                            "severity": sev_level.value,
                            "severity_score": sev_score,
                            "x_min": det["x_min"],
                            "y_min": det["y_min"],
                            "x_max": det["x_max"],
                            "y_max": det["y_max"],
                            "distance_meters": distance_est
                        }
                        frame_detections.append(det_record)
                        all_detections_list.append(det_record)
                except Exception as db_err:
                    print(f"⚠️ [Frame Persistence Warning]: {db_err}")
                    # Still record in memory for video report & telemetry
                    for det in raw_detections:
                        sev_level, sev_score = SeverityAnalysisService.calculate_detection_severity(
                            det,
                            frame_width=processor.width,
                            frame_height=processor.height,
                            cluster_count=len(raw_detections)
                        )
                        cat_val = det.get("category", "POTHOLE")
                        cat_str = cat_val.value if hasattr(cat_val, "value") else str(cat_val)
                        distance_est = SeverityAnalysisService.estimate_perspective_distance(
                            det,
                            frame_height=processor.height
                        )
                        det_record = {
                            "category": cat_str,
                            "confidence": det["confidence"],
                            "severity": sev_level.value,
                            "severity_score": sev_score,
                            "x_min": det["x_min"],
                            "y_min": det["y_min"],
                            "x_max": det["x_max"],
                            "y_max": det["y_max"],
                            "distance_meters": distance_est
                        }
                        frame_detections.append(det_record)
                        all_detections_list.append(det_record)

                # Annotate Frame with bounding boxes
                annotated_img = raw_frame.copy()
                for d in frame_detections:
                    color = (0, 0, 255) if d["severity"] == "critical" else (0, 165, 255) if d["severity"] == "high" else (0, 255, 0)
                    cv2.rectangle(
                        annotated_img,
                        (int(d["x_min"]), int(d["y_min"])),
                        (int(d["x_max"]), int(d["y_max"])),
                        color,
                        2
                    )
                    label = f"{d['category']} {int(d['confidence']*100)}%"
                    cv2.putText(
                        annotated_img,
                        label,
                        (int(d["x_min"]), max(15, int(d["y_min"]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                if video_writer:
                    video_writer.write(annotated_img)

                # Encode Frame to Base64 for Live UI Stream
                _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                base64_str = base64.b64encode(buffer).decode('utf-8')
                frame_base64 = f"data:image/jpeg;base64,{base64_str}"

                formatted_detections = []
                for det in frame_detections:
                    formatted_detections.append({
                        "category": det["category"],
                        "confidence": round(float(det["confidence"]), 2),
                        "severity": det["severity"].upper(),
                        "x_min": int(det["x_min"]),
                        "y_min": int(det["y_min"]),
                        "x_max": int(det["x_max"]),
                        "y_max": int(det["y_max"])
                    })

                base_lat = 28.4595 + (frame_num * 0.00008)
                base_lon = 77.0266 + (frame_num * 0.00009)
                live_road_health = round(max(20.0, 100.0 - (len(all_detections_list) * 3.5)), 1)

                ws_frame_msg = {
                    "type": "frame",
                    "video_id": video.id,
                    "frame_number": frame_num,
                    "timestamp": round(float(timestamp_sec), 2),
                    "elapsed_time": round(float(timestamp_sec), 2),
                    "fps": round(float(processor.fps or 30.0), 1),
                    "progress": current_progress,
                    "image_data": frame_base64,
                    "image_base64": base64_str,
                    "image_url": frame_base64,
                    "road_health": live_road_health,
                    "gps": {
                        "latitude": round(base_lat, 6),
                        "longitude": round(base_lon, 6)
                    },
                    "detections": formatted_detections
                }
                await ws_manager.broadcast(ws_frame_msg)
                await ws_broadcaster.broadcast(ws_frame_msg)

                del annotated_img, raw_frame, preprocessed_frame, buffer
                await asyncio.sleep(0.01)

            if video_writer:
                video_writer.release()
                video.processed_file_path = output_mp4_path

            processor.close()

            await send_ws_update("Generating Report", 80, "FastAPI WS: Computing Road Health Index & GPS coordinates...")

            # Compute Road Health Index
            analytics_res = SeverityAnalysisService.calculate_road_health_index(
                all_detections_list,
                video_duration_seconds=video.duration_seconds
            )

            # Generate GPS Trajectory
            trajectory = GPSExtractionService.generate_interpolated_trajectory(
                total_frames=video.total_frames,
                fps=video.fps
            )
            for point in trajectory:
                db_gps = GPSData(
                    video_id=video.id,
                    frame_number=point["frame_number"],
                    latitude=point["latitude"],
                    longitude=point["longitude"],
                    altitude_meters=point["altitude_meters"],
                    speed_kmh=point["speed_kmh"],
                    road_name=point["road_name"]
                )
                db.add(db_gps)

            # ORM Analytics Record
            db_analytics = RoadAnalytics(
                video_id=video.id,
                road_health_score=analytics_res["road_health_score"],
                total_detections=analytics_res["total_detections"],
                pothole_count=analytics_res["pothole_count"],
                crack_count=analytics_res["crack_count"],
                critical_count=analytics_res["critical_count"],
                damage_density_per_km=analytics_res["damage_density_per_km"],
                overall_severity=analytics_res["overall_severity"]
            )
            db.add(db_analytics)

            await send_ws_update("Saving Results", 95, f"FastAPI WS: Persisting {len(all_detections_list)} damage detections to database...")

            video.status = ProcessingStatus.COMPLETED
            await db.commit()

            processing_progress_state["is_processing"] = False
            processing_progress_state["progress_percent"] = 100
            processing_progress_state["status"] = "Completed"

            await send_ws_update("Finished", 100, "FastAPI WS: Processing pipeline finished successfully!")
            finished_msg = {
                "type": "finished",
                "video_id": video.id,
                "progress": 100,
                "message": "AI Processing pipeline completed successfully."
            }
            await ws_manager.broadcast(finished_msg)
            await ws_broadcaster.broadcast(finished_msg)

        except Exception as e:
            print(f"[Processing Pipeline Error]: {e}")
            if video:
                try:
                    video.status = ProcessingStatus.FAILED
                    await db.commit()
                except Exception:
                    pass
            await send_ws_update("Finished", 0, f"FastAPI WS Error: {str(e)}")


@router.post("/run", response_model=ProcessVideoResponse)
async def process_video_pipeline(
    req: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger AI Computer Vision Processing on an uploaded video.
    Launches asynchronous OpenCV frame decoding, YOLOv11 inference, severity scoring,
    and real-time WebSocket telemetry streaming.
    """
    target_video_id = req.video_id
    if not target_video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required field: 'video_id'"
        )

    stmt = select(Video).where(Video.id == target_video_id)
    video = (await db.execute(stmt)).scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{target_video_id}' not found."
        )

    # Mark video status as processing
    video.status = ProcessingStatus.PROCESSING
    await db.commit()

    # Launch background processing pipeline asynchronously
    background_tasks.add_task(
        execute_video_processing_task,
        video_id=video.id,
        confidence_threshold=float(req.confidence_threshold if req.confidence_threshold is not None else 0.35),
        frame_skip=int(req.frame_skip if req.frame_skip is not None else 5),
        enable_histogram_equalization=bool(req.enable_histogram_equalization if req.enable_histogram_equalization is not None else True),
        enable_gaussian_blur=bool(req.enable_gaussian_blur if req.enable_gaussian_blur is not None else True)
    )

    return {
        "video_id": video.id,
        "status": ProcessingStatus.PROCESSING,
        "message": "AI Computer Vision processing pipeline started successfully in background.",
        "total_frames_processed": 0,
        "total_detections_found": 0,
        "road_health_score": 100.0
    }


@router.get("/status")
async def get_pipeline_processing_status():
    """
    Returns the live status of the AI video processing pipeline.
    """
    return {
        "is_processing": processing_progress_state.get("is_processing", False),
        "video_id": processing_progress_state.get("video_id"),
        "current_frame": processing_progress_state.get("current_frame", 0),
        "total_frames": processing_progress_state.get("total_frames", 0),
        "progress_percent": processing_progress_state.get("progress_percent", 0),
        "current_fps": processing_progress_state.get("current_fps", 30.0),
        "estimated_time_remaining_sec": processing_progress_state.get("estimated_time_remaining_sec", 0),
        "status": processing_progress_state.get("status", "idle")
    }

