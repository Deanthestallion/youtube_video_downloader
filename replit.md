# AI-Powered Video Processing Application

## Overview

This is a Flask-based web application that provides comprehensive AI-powered video processing capabilities. The application combines silence removal, face detection and zoom, multi-language transcription, and subtitle creation in the original detected language to create professional video content optimized for mobile viewing. It features a user-friendly interface for uploading videos, real-time processing status updates, and multiple download options for processed content.

## System Architecture

The application follows a simple web architecture pattern:

- **Frontend**: HTML templates with Bootstrap for responsive UI, JavaScript for client-side interactions
- **Backend**: Flask web framework serving both API endpoints and web pages
- **Processing Engine**: Integration with `auto-editor` for silence removal and `ffmpeg` for video manipulation
- **File Storage**: Local filesystem for temporary file storage during processing
- **Deployment**: Gunicorn WSGI server with autoscale deployment target

## Key Components

### Web Application Layer
- **Flask App** (`app.py`): Main application server handling file uploads, processing coordination, and serving results
- **Templates**: HTML templates using Bootstrap dark theme for consistent UI
  - `index.html`: Upload interface with drag-and-drop functionality
  - `result.html`: Processing status and download interface
- **Static Assets**: CSS and JavaScript files for enhanced user experience

### AI-Powered Video Processing Pipeline
- **Auto-editor Integration**: Subprocess calls to `auto-editor` for silence removal
- **Face Detection & Zoom**: OpenCV-based face detection with automatic cropping for mobile optimization
- **OpenAI Whisper Integration**: Speech-to-text transcription using OpenAI's Whisper model
- **Subtitle Generation**: Automatic creation of SRT subtitle files with precise timing
- **FFmpeg Integration**: Video cropping, thumbnail extraction, audio extraction, and subtitle burning
- **Async Processing**: Threading-based background processing to handle long-running AI operations
- **Status Tracking**: Multi-stage processing status with detailed progress indicators

### File Management
- **Upload Handling**: Secure file upload with size validation (500MB limit)
- **Temporary Storage**: Organized directory structure for inputs, outputs, and thumbnails
- **Unique Identifiers**: UUID-based file naming to prevent conflicts

## Data Flow

1. **Upload Phase**: User selects video file through web interface
2. **Validation**: Client-side file size and type validation
3. **Processing Initiation**: Server receives file, assigns UUID, starts background processing
4. **Silence Removal**: Auto-editor processes video to remove silent segments first
5. **Face Detection**: OpenCV analyzes video frames to detect face regions
6. **Video Cropping**: FFmpeg crops video to zoom in on detected face area for mobile optimization
7. **Audio Extraction**: FFmpeg extracts audio track from processed video for transcription
8. **Language Detection**: OpenAI Whisper analyzes audio and detects the spoken language automatically
9. **AI Transcription**: OpenAI Whisper transcribes speech from final audio with word-level timestamps
10. **Subtitle Creation**: System generates SRT subtitle file in original detected language with precise timing
11. **Subtitle Burning**: FFmpeg burns subtitles in original language directly into the processed video
12. **Status Updates**: Real-time multi-stage status updates via polling mechanism
13. **Download**: Processed video with face zoom, perfectly synced subtitles in original language, SRT subtitle file, and raw transcript text file available for download

## External Dependencies

### Core Dependencies
- **Flask**: Web framework and request handling
- **OpenAI**: API client for Whisper transcription services
- **PySRT**: SRT subtitle file creation and manipulation
- **OpenCV**: Video processing and computer vision tasks
- **Gunicorn**: Production WSGI server
- **Auto-editor**: Third-party tool for automated silence removal
- **FFmpeg**: Video encoding, decoding, subtitle burning, and audio extraction

### System Dependencies
- **Python 3.11**: Runtime environment
- **PostgreSQL**: Database support (available but not currently used)
- **OpenSSL**: Secure communications
- **OpenGL**: Graphics rendering support

### Frontend Dependencies
- **Bootstrap**: UI framework with dark theme
- **Font Awesome**: Icon library
- **Custom CSS/JS**: Enhanced user interactions and styling

## Deployment Strategy

The application is configured for Replit deployment with the following characteristics:

- **Deployment Target**: Autoscale for handling variable load
- **Runtime**: Python 3.11 with Nix package management
- **Server**: Gunicorn with port binding to 5000
- **Process Management**: Parallel workflow execution
- **File Persistence**: Local file system (suitable for temporary processing files)

### API Usage

### Web Interface
The web interface now requires users to provide their OpenAI API key directly in the upload form. The key is processed securely per-request without storage.

### API Endpoint
Use the `/api/process` endpoint for programmatic access:

```bash
# Upload video with API key via form data
curl -X POST "https://your-app.replit.app/api/process" \
  -F "file=@video.mp4" \
  -F "api_key=sk-your-openai-api-key"

# Check processing status
curl "https://your-app.replit.app/status/{uid}"

# Download processed video
curl "https://your-app.replit.app/download/output_{uid}.mp4" -o processed_video.mp4

# Download subtitles in original language
curl "https://your-app.replit.app/download-subtitle/{uid}" -o subtitles.srt

# Download raw transcript text
curl "https://your-app.replit.app/download-transcript/{uid}" -o transcript.txt
```

### Response Format
```json
{
  "message": "Processing with transcription and subtitles started",
  "uid": "unique-id",
  "status_url": "/status/unique-id",
  "video_download_url": "/download/output_unique-id.mp4",
  "subtitle_download_url": "/download-subtitle/unique-id",
  "transcript_download_url": "/download-transcript/unique-id",
  "thumbnail_image_url": "/thumbnail/thumb_unique-id.jpg",
  "thumbnail_video_url": "/thumbnail/thumbclip_unique-id.mp4"
}
```

## Production Considerations
- Files are stored locally, which works for temporary processing but may need cloud storage for persistence
- In-memory status tracking will reset on server restart
- No database integration currently implemented despite PostgreSQL availability
- API keys are processed per-request and never stored on the server

## Changelog

- June 23, 2025: Raw transcript download feature
  - Added plain text transcript file creation alongside SRT subtitles
  - Implemented download endpoint for raw Whisper transcript output
  - Enhanced result page to show both subtitle and transcript download options
  - Improved content accessibility with multiple format options

- June 23, 2025: Original language subtitle preservation
  - Modified subtitle creation to preserve original detected language instead of translating
  - Updated UI messaging to reflect subtitles in original language
  - Maintained language detection capability while keeping authentic content
  - Enhanced user experience by respecting original language content

- June 23, 2025: Dynamic OpenAI API key support
  - Removed dependency on environment-stored OPENAI_API_KEY
  - Added dynamic API key acceptance through HTTP requests
  - Updated web form to include API key input field
  - Enhanced API endpoint to accept API keys via form data or JSON
  - Improved security by processing API keys per-request without storage

- June 23, 2025: Language detection and transcription feature
  - Added automatic language detection using OpenAI Whisper
  - Implemented multi-language transcription with word-level timestamps
  - Enhanced subtitle creation for any detected language
  - Updated UI to show language detection and transcription status

- June 19, 2025: Face detection and zoom feature
  - Added OpenCV-based face detection for automatic video cropping
  - Implemented intelligent zoom functionality to focus on faces for better mobile viewing
  - Updated processing pipeline: silence removal → face detection → zoom → transcription → subtitles
  - Enhanced UI with face detection status indicator
  - Improved mobile optimization with automatic square crop around detected faces

- June 19, 2025: Processing pipeline optimization
  - Improved subtitle timing accuracy by transcribing after silence removal
  - Updated processing order: silence removal → audio extraction → transcription → subtitle creation → burning
  - Enhanced status messages to reflect improved workflow
  - Fixed subtitle generation compatibility with OpenAI Whisper response format

- June 19, 2025: Major AI enhancement update
  - Added OpenAI Whisper integration for speech transcription
  - Implemented automatic SRT subtitle generation with word-level timing
  - Added subtitle burning functionality using FFmpeg
  - Enhanced UI with multi-stage processing status indicators
  - Added separate download endpoints for videos and subtitle files
  - Updated project description to reflect AI-powered capabilities

- June 19, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.