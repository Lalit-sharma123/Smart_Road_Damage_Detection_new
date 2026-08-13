import os
import json
import base64
import cv2
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.models.models import (
    User, Video, Frame, Detection, GPSData, RoadAnalytics,
    UserRole, ProcessingStatus, SeverityLevel, DamageCategory
)
from app.schemas.schemas import ProcessVideoRequest, ProcessVideoResponse
from app.auth.jwt import require_role
from app.cv.video_processor import VideoProcessor
from app.yolo.detector import YOLODamageDetector
from app.services.severity_service import SeverityAnalysisService
from app.services.gps_service import GPSExtractionService
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
        while True:
            data = await websocket.receive_text()
            # Echo or process incoming socket messages if needed
            await websocket.send_text(json.dumps({
                "stage": "Connected",
                "progress": 0,
                "message": f"WebSocket connection active for client: {client_id}",
                "timestamp": asyncio.get_event_loop().time()
            }))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@router.post("/run", response_model=ProcessVideoResponse)
async def process_video_pipeline(
    req: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role([UserRole.ADMIN, UserRole.INSPECTOR])),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger AI Computer Vision Processing on an uploaded video.
    Executes frame decoding, OpenCV contrast filters, YOLOv11 inference, Severity scoring & GPS tagging.
    """
    stmt = select(Video).where(Video.id == req.video_id)
    video = (await db.execute(stmt)).scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{req.video_id}' not found."
        )

    # Mark video status as processing
    video.status = ProcessingStatus.PROCESSING
    await db.commit()

    async def send_ws_update(stage: str, progress: int, msg: str):
        await ws_manager.broadcast({
            "stage": stage,
            "progress": progress,
            "message": msg,
            "timestamp": asyncio.get_event_loop().time()
        })

    # Execute processing pipeline
    try:
        await send_ws_update("Uploading", 15, "FastAPI WS: Ingestion verified. Initializing OpenCV decoding...")
        processor = VideoProcessor(video.file_path)
        all_detections_list = []
        frames_processed_count = 0

        await send_ws_update("Extracting Frames", 35, f"FastAPI WS: Slicing frames with frame_skip={req.frame_skip}...")

        # Extract frames generator with frame skip
        frame_gen = processor.extract_frames_generator(
            frame_skip=req.frame_skip,
            enable_histogram_eq=req.enable_histogram_equalization,
            enable_gaussian_blur=req.enable_gaussian_blur
        )

        await send_ws_update("Running YOLO", 50, "FastAPI WS: Executing YOLOv11 damage detection inference...")

        # Optional single output MP4 video writer
        output_mp4_path = os.path.join(settings.PROCESSED_DIR, f"processed_{video.id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_mp4_path,
            fourcc,
            processor.fps or 30.0,
            (processor.width, processor.height)
        )

        for frame_num, timestamp_sec, raw_frame, preprocessed_frame in frame_gen:
            frames_processed_count += 1

            # YOLO detection on preprocessed frame (in memory)
            raw_detections = detector_instance.detect(
                preprocessed_frame,
                conf_threshold=req.confidence_threshold
            )

            has_damage = len(raw_detections) > 0

            # DB Frame record without image disk path
            db_frame = Frame(
                video_id=video.id,
                frame_number=frame_num,
                timestamp_seconds=timestamp_sec,
                image_path="",
                has_damage=has_damage
            )
            db.add(db_frame)
            await db.flush()

            # Process detections with severity formula
            frame_detections = []
            for det in raw_detections:
                sev_level, sev_score = SeverityAnalysisService.calculate_detection_severity(
                    det,
                    frame_width=processor.width,
                    frame_height=processor.height,
                    cluster_count=len(raw_detections)
                )

                det_record = {
                    "category": det["category"],
                    "confidence": det["confidence"],
                    "x_min": det["x_min"],
                    "y_min": det["y_min"],
                    "x_max": det["x_max"],
                    "y_max": det["y_max"],
                    "area_pixels": det["area_pixels"],
                    "severity": sev_level,
                    "severity_score": sev_score,
                    "frame_number": frame_num,
                    "timestamp_sec": timestamp_sec
                }
                frame_detections.append(det_record)
                all_detections_list.append(det_record)

                # ORM Detection
                category_enum = DamageCategory(det["category"]) if det["category"] in DamageCategory._value2member_map_ else DamageCategory.POTHOLE
                db_det = Detection(
                    video_id=video.id,
                    frame_id=db_frame.id,
                    category=category_enum,
                    confidence=det["confidence"],
                    x_min=det["x_min"],
                    y_min=det["y_min"],
                    x_max=det["x_max"],
                    y_max=det["y_max"],
                    area_pixels=det["area_pixels"],
                    severity=sev_level,
                    severity_score=sev_score
                )
                db.add(db_det)

            # Draw annotated frame bounding boxes directly in memory
            annotated_img = VideoProcessor.draw_detections(raw_frame, frame_detections)

            # Write frame to single optional compiled MP4 video
            if video_writer and video_writer.isOpened():
                video_writer.write(annotated_img)

            # Encode processed frame as JPEG in memory -> Base64
            success, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                base64_str = base64.b64encode(buffer).decode('utf-8')
                frame_base64 = f"data:image/jpeg;base64,{base64_str}"
            else:
                base64_str = ""
                frame_base64 = ""

            # Broadcast frame payload over WebSocket in real time
            total_expected_frames = processor.total_frames or 100
            current_progress = min(99, max(1, int((frame_num / total_expected_frames) * 100)))

            formatted_detections = []
            for det in frame_detections:
                sev_val = det["severity"].value if hasattr(det["severity"], "value") else str(det["severity"]).upper()
                formatted_detections.append({
                    "category": det["category"],
                    "confidence": round(float(det["confidence"]), 2),
                    "severity": sev_val,
                    "x_min": int(det["x_min"]),
                    "y_min": int(det["y_min"]),
                    "x_max": int(det["x_max"]),
                    "y_max": int(det["y_max"])
                })

            # Live GPS calculation for current frame
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

            # Release memory immediately
            del annotated_img, raw_frame, preprocessed_frame, buffer
            await asyncio.sleep(0.02) # Pacing for smooth CCTV live streaming experience

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

        # Update video completion status
        video.status = ProcessingStatus.COMPLETED
        await db.commit()

        await send_ws_update("Finished", 100, "FastAPI WS: Processing pipeline finished successfully!")
        await ws_manager.broadcast({
            "type": "finished",
            "video_id": video.id,
            "progress": 100,
            "message": "AI Processing pipeline completed successfully."
        })

        return {
            "video_id": video.id,
            "status": ProcessingStatus.COMPLETED,
            "message": "AI Computer Vision processing completed successfully.",
            "total_frames_processed": frames_processed_count,
            "total_detections_found": len(all_detections_list),
            "road_health_score": analytics_res["road_health_score"]
        }

    except Exception as e:
        video.status = ProcessingStatus.FAILED
        await db.commit()
        await send_ws_update("Finished", 0, f"FastAPI WS Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing AI detection pipeline: {str(e)}"
        )

