import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config.config import settings
from app.database.database import init_db
from app.api.auth import router as auth_router
from app.api.videos import router as video_router
from app.api.process import router as process_router
from app.api.reports import router as report_router
from app.api.analytics import router as analytics_router
from app.api.dashboard import router as dashboard_router
from app.api.cameras import router as cameras_router
from app.api.models_api import router as models_router
from app.api.users_api import router as users_router
from app.api.logs import router as logs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifecycle Context Manager.
    Initializes PostgreSQL tables on startup.
    """
    print("Initializing Smart Road Damage Detection Database Schema...")
    try:
        await init_db()
        print("Database schema successfully synchronized.")
    except Exception as e:
        print(f"Database initialization note: {e}. Ensure PostgreSQL is running.")
    yield
    print("Shutting down Smart Road Damage Backend Application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    ## Production Backend API for Smart Road Damage Detection and Analysis System

    ### Core Capabilities:
    * **JWT Authentication & RBAC**: Admin, Inspector, Viewer roles.
    * **Video Ingestion & Processing**: OpenCV frame extraction, frame skipping, CLAHE histogram equalization, Gaussian blur.
    * **YOLOv11 Detection**: Deep learning bounding box localization for Potholes, Cracks, Missing Asphalt, Broken Road.
    * **Severity Formula Engine**: Area ratio, confidence, density, and class hazard weighting into Low, Medium, High, Critical ratings.
    * **GPS Telemetry Integration**: Geotagging and route trajectory plotting.
    * **Automated Audit Reports**: PDF, Excel (.xlsx), and CSV exports.
    * **Analytics & Road Health Index**: Section quality scoring (0-100) and defect density tracking.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Execution Time & Middleware Logger
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An internal server error occurred.",
            "detail": str(exc),
            "path": request.url.path
        },
    )


# Mount Static Media Files (Uploads, Processed frames, Reports)
app.mount("/processed", StaticFiles(directory=settings.PROCESSED_DIR), name="processed_direct")
app.mount("/static/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/static/processed", StaticFiles(directory=settings.PROCESSED_DIR), name="processed")
app.mount("/static/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# Include API Routers under /api/v1
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(video_router, prefix=settings.API_V1_STR)
app.include_router(process_router, prefix=settings.API_V1_STR)
app.include_router(report_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(cameras_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(users_router, prefix=settings.API_V1_STR)
app.include_router(logs_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Health Check"])
async def root_health_check():
    """System Health Check Endpoint."""
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": time.time(),
        "docs": "/docs"
    }
