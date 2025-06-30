from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
import os
import uuid
import threading
import subprocess
import cv2
import logging
import json
from openai import OpenAI
import pysrt
import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

OUTPUT_DIR = "outputs"
THUMBNAIL_DIR = "thumbnails"
SUBTITLE_DIR = "subtitles"
TRANSCRIPT_DIR = "transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(SUBTITLE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# OpenAI client will be initialized per request with dynamic API key
print("[INFO] OpenAI API key will be provided dynamically via HTTP requests")

# Store processing status
processing_status = {}

def run_auto_editor(input_path, output_path, uid):
    """Run auto-editor to remove silent scenes from video"""
    try:
        print(f"[INFO] Starting auto-editor for {input_path}")
        processing_status[uid] = "processing"
        subprocess.run([
            'auto-editor', input_path,
            '--output_file', output_path,
            '--no-open'
        ], check=True)
        processing_status[uid] = "completed"
        print(f"[INFO] Finished processing {output_path}")
    except subprocess.CalledProcessError as e:
        processing_status[uid] = "failed"
        print(f"[ERROR] auto-editor failed: {e}")
    except Exception as e:
        processing_status[uid] = "failed"
        print(f"[ERROR] Unexpected error in auto-editor: {e}")

def extract_video_thumbnail(input_path, output_path, start=0, duration=5):
    """Extract a short video clip as thumbnail"""
    try:
        subprocess.run([
            'ffmpeg',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',
            '-y',  # Overwrite output file
            output_path
        ], check=True)
        print(f"[INFO] Video thumbnail created at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e}")

def extract_audio_for_transcription(input_path, output_path):
    """Extract audio from video for transcription"""
    try:
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-vn',  # No video
            '-acodec', 'mp3',
            '-ar', '16000',  # 16kHz sample rate for Whisper
            '-ac', '1',  # Mono
            '-y',
            output_path
        ], check=True)
        print(f"[INFO] Audio extracted to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        return False

def transcribe_with_whisper(audio_path, api_key):
    """Transcribe audio using OpenAI Whisper with language detection"""
    if not api_key:
        print("[ERROR] OpenAI API key not provided")
        return None, None
    
    try:
        openai_client = OpenAI(api_key=api_key)
        
        with open(audio_path, "rb") as audio_file:
            # First, detect the language
            language_detection = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            
            detected_language = getattr(language_detection, 'language', 'unknown')
            print(f"[INFO] Detected language: {detected_language}")
            
        # Now transcribe with word-level timestamps
        with open(audio_path, "rb") as audio_file:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
        print(f"[INFO] Transcription completed in {detected_language}")
        return transcript, detected_language
        
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return None, None

def translate_to_english(text, source_language, api_key):
    """Translate text to English using OpenAI GPT"""
    if not api_key:
        print("[ERROR] OpenAI API key not provided")
        return text
    
    # If already in English, return as-is
    if source_language.lower() in ['en', 'english']:
        return text
    
    try:
        openai_client = OpenAI(api_key=api_key)
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the following {source_language} text to natural, fluent English. Maintain the original tone and meaning. Return only the translation without any explanations."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        
        translated_text = response.choices[0].message.content.strip()
        print(f"[INFO] Translation completed from {source_language} to English")
        return translated_text
        
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return text  # Return original text if translation fails

def create_srt_subtitles(transcript, srt_path, detected_language=None, api_key=None):
    """Create SRT subtitle file from Whisper transcript in original language"""
    try:
        subs = pysrt.SubRipFile()
        
        # Check if we have word-level timestamps
        if hasattr(transcript, 'words') and transcript.words:
            # Group words into subtitle segments (max 8 words per subtitle)
            words_per_subtitle = 8
            words = transcript.words
            
            for i in range(0, len(words), words_per_subtitle):
                word_group = words[i:i + words_per_subtitle]
                
                # Get start and end times - handle both dict and object formats
                if hasattr(word_group[0], 'start'):
                    start_time = word_group[0].start
                    end_time = word_group[-1].end
                    subtitle_text = ' '.join([word.word for word in word_group])
                else:
                    start_time = word_group[0]['start']
                    end_time = word_group[-1]['end']
                    subtitle_text = ' '.join([word['word'] for word in word_group])
                
                # Keep original language - no translation
                # subtitle_text remains in the detected language
                
                # Convert to SRT time format
                start_srt = pysrt.SubRipTime(seconds=start_time)
                end_srt = pysrt.SubRipTime(seconds=end_time)
                
                # Create subtitle item
                sub = pysrt.SubRipItem(
                    index=len(subs) + 1,
                    start=start_srt,
                    end=end_srt,
                    text=subtitle_text
                )
                subs.append(sub)
        
        # If no word-level timing, use segments or create basic subtitles
        elif hasattr(transcript, 'segments') and transcript.segments:
            for segment in transcript.segments:
                start_time = segment.start if hasattr(segment, 'start') else segment['start']
                end_time = segment.end if hasattr(segment, 'end') else segment['end']
                text = segment.text if hasattr(segment, 'text') else segment['text']
                
                # Keep original language - no translation needed
                # text remains in the detected language
                
                sub = pysrt.SubRipItem(
                    index=len(subs) + 1,
                    start=pysrt.SubRipTime(seconds=start_time),
                    end=pysrt.SubRipTime(seconds=end_time),
                    text=text.strip() if text else ""
                )
                subs.append(sub)
        
        else:
            # Fallback: create subtitle with full text split into chunks
            full_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
            
            # Keep original language - no translation needed
            # full_text remains in the detected language
            
            words = full_text.split()
            words_per_chunk = 8
            duration_per_chunk = 3  # 3 seconds per chunk
            
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                chunk_text = ' '.join(chunk_words)
                
                start_time = i // words_per_chunk * duration_per_chunk
                end_time = start_time + duration_per_chunk
                
                sub = pysrt.SubRipItem(
                    index=len(subs) + 1,
                    start=pysrt.SubRipTime(seconds=start_time),
                    end=pysrt.SubRipTime(seconds=end_time),
                    text=chunk_text
                )
                subs.append(sub)
        
        # Save SRT file
        subs.save(srt_path, encoding='utf-8')
        language_info = f" in {detected_language}" if detected_language else ""
        print(f"[INFO] SRT subtitles created at {srt_path} with {len(subs)} entries{language_info}")
        return True
    except Exception as e:
        print(f"[ERROR] SRT creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_face_region(video_path):
    """Detect face region in video and return crop parameters"""
    cap = None
    try:
        # Load OpenCV face cascade - use fallback path if needed
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            # Fallback - create a simple center crop if cascade loading fails
            print("[WARNING] Could not load face cascade, using center crop")
            cap = cv2.VideoCapture(video_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            crop_size = min(video_width, video_height)
            x = (video_width - crop_size) // 2
            y = (video_height - crop_size) // 2
            return x, y, crop_size, crop_size
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample frames to detect face
        sample_frames = min(30, total_frames // 10) if total_frames > 10 else total_frames
        face_detections = []
        
        for i in range(sample_frames):
            frame_number = i * (total_frames // sample_frames) if sample_frames > 0 else i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Take the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    face_detections.append(largest_face)
        
        cap.release()
        
        if not face_detections:
            print("[INFO] No faces detected, using center crop")
            # Default to center crop if no face detected
            crop_size = min(video_width, video_height)
            x = (video_width - crop_size) // 2
            y = (video_height - crop_size) // 2
            return x, y, crop_size, crop_size
        
        # Calculate average face position and size
        avg_x = sum(face[0] for face in face_detections) // len(face_detections)
        avg_y = sum(face[1] for face in face_detections) // len(face_detections)
        avg_w = sum(face[2] for face in face_detections) // len(face_detections)
        avg_h = sum(face[3] for face in face_detections) // len(face_detections)
        
        # Calculate face center
        face_center_x = avg_x + avg_w // 2
        face_center_y = avg_y + avg_h // 2
        
        # Determine crop size (make it reasonable relative to video size)
        face_size = max(avg_w, avg_h)
        crop_size = min(face_size * 4, min(video_width, video_height))  # 4x face size, but not larger than video
        crop_size = max(crop_size, 200)  # Minimum crop size
        
        # Center crop around face, ensuring boundaries
        crop_x = max(0, face_center_x - crop_size // 2)
        crop_y = max(0, face_center_y - crop_size // 2)
        
        # Adjust if crop goes outside video boundaries
        if crop_x + crop_size > video_width:
            crop_x = video_width - crop_size
        if crop_y + crop_size > video_height:
            crop_y = video_height - crop_size
            
        # Final boundary check - ensure all values are positive and within bounds
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_size = min(crop_size, video_width - crop_x, video_height - crop_y)
        
        # Ensure crop size is reasonable and positive
        if crop_size <= 0:
            print("[WARNING] Invalid crop size, using center crop")
            crop_size = min(video_width, video_height) // 2
            crop_x = (video_width - crop_size) // 2
            crop_y = (video_height - crop_size) // 2
        
        print(f"[INFO] Face detected, crop region: {crop_x}, {crop_y}, {crop_size}x{crop_size} (video: {video_width}x{video_height})")
        return crop_x, crop_y, crop_size, crop_size
        
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        # Fallback to center crop - get video dimensions
        try:
            if cap is None:
                cap = cv2.VideoCapture(video_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if cap:
                cap.release()
        except:
            video_width, video_height = 640, 480  # Safe defaults
        
        crop_size = min(video_width, video_height)
        x = (video_width - crop_size) // 2
        y = (video_height - crop_size) // 2
        return x, y, crop_size, crop_size

def apply_face_zoom(input_path, output_path):
    """Apply face detection and zoom/crop to video"""
    try:
        # Detect face region
        crop_x, crop_y, crop_w, crop_h = detect_face_region(input_path)
        
        # Apply crop using FFmpeg
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-vf', f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y}',
            '-c:a', 'copy',
            '-y',
            output_path
        ], check=True)
        
        print(f"[INFO] Face zoom applied to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Face zoom failed: {e}")
        return False

def burn_subtitles_to_video(input_path, srt_path, output_path):
    """Burn subtitles into video using FFmpeg"""
    try:
        # Escape the subtitle path for FFmpeg
        srt_path_escaped = srt_path.replace(':', '\\:').replace(',', '\\,')
        
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-vf', f"subtitles='{srt_path_escaped}':force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,BorderStyle=1,Outline=2'",
            '-c:a', 'copy',
            '-y',
            output_path
        ], check=True)
        print(f"[INFO] Subtitles burned into video at {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subtitle burning failed: {e}")
        return False

def process_video_with_subtitles(input_path, output_path, uid, api_key=None):
    """Complete video processing with silence removal, face zoom, transcription and subtitles"""
    try:
        processing_status[uid] = "removing_silence"
        
        # First, run auto-editor for silence removal
        temp_edited_path = os.path.join(OUTPUT_DIR, f"temp_edited_{uid}.mp4")
        try:
            subprocess.run([
                'auto-editor', input_path,
                '--output_file', temp_edited_path,
                '--no-open'
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] auto-editor failed: {e}")
            processing_status[uid] = "failed"
            return
        
        processing_status[uid] = "detecting_face"
        
        # Apply face detection and zoom
        temp_zoomed_path = os.path.join(OUTPUT_DIR, f"temp_zoomed_{uid}.mp4")
        if not apply_face_zoom(temp_edited_path, temp_zoomed_path):
            processing_status[uid] = "failed"
            return
        
        processing_status[uid] = "extracting_audio"
        
        # Extract audio from the zoomed video for transcription
        audio_path = os.path.join(OUTPUT_DIR, f"audio_{uid}.mp3")
        if not extract_audio_for_transcription(temp_zoomed_path, audio_path):
            processing_status[uid] = "failed"
            return
        
        processing_status[uid] = "transcribing"
        
        # Transcribe the edited audio with Whisper and detect language
        result = transcribe_with_whisper(audio_path, api_key)
        if not result or len(result) != 2:
            processing_status[uid] = "failed"
            return
        
        transcript, detected_language = result
        if not transcript:
            processing_status[uid] = "failed"
            return
        
        processing_status[uid] = "creating_subtitles"
        
        # Save raw transcript text
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"transcript_{uid}.txt")
        try:
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # Write transcript with language info
                f.write(f"Language detected: {detected_language}\n")
                f.write(f"Transcript:\n\n")
                f.write(transcript.text if hasattr(transcript, 'text') else str(transcript))
            print(f"[INFO] Raw transcript saved at {transcript_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save transcript: {e}")
        
        # Create SRT subtitles in original language
        srt_path = os.path.join(SUBTITLE_DIR, f"subtitles_{uid}.srt")
        if not create_srt_subtitles(transcript, srt_path, detected_language, api_key):
            processing_status[uid] = "failed"
            return
        
        processing_status[uid] = "burning_subtitles"
        
        # Burn subtitles into the final video
        if not burn_subtitles_to_video(temp_zoomed_path, srt_path, output_path):
            processing_status[uid] = "failed"
            return
        
        # Clean up temporary files
        try:
            os.remove(audio_path)
            os.remove(temp_edited_path)
            os.remove(temp_zoomed_path)
        except:
            pass  # Don't fail if cleanup fails
        
        processing_status[uid] = "completed"
        print(f"[INFO] Complete processing finished for {output_path}")
        
    except Exception as e:
        processing_status[uid] = "failed"
        print(f"[ERROR] Complete processing failed: {e}")

@app.route("/", methods=["GET"])
def index():
    """Main page with upload form"""
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_video():
    """Handle video upload and start processing"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not file.filename or not str(file.filename).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        flash('Please upload a valid video file (.mp4, .avi, .mov, .mkv, .webm)', 'error')
        return redirect(url_for('index'))

    # Get OpenAI API key from form
    api_key = request.form.get('api_key', '').strip()
    if not api_key:
        flash('OpenAI API key is required for transcription and translation features', 'error')
        return redirect(url_for('index'))

    uid = uuid.uuid4().hex
    input_path = os.path.join(OUTPUT_DIR, f"input_{uid}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"output_{uid}.mp4")
    image_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumb_{uid}.jpg")
    video_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumbclip_{uid}.mp4")

    try:
        file.save(input_path)
        print(f"[INFO] File saved to {input_path}")

        # Extract image thumbnail from middle of video
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_thumb_path, frame)
                print(f"[INFO] Image thumbnail saved to {image_thumb_path}")
        cap.release()

        # Extract video thumbnail (5-second clip starting from 5 seconds)
        extract_video_thumbnail(input_path, video_thumb_path, start=5, duration=6)

        # Initialize processing status
        processing_status[uid] = "starting"

        # Start full processing with subtitles in background with API key
        threading.Thread(target=process_video_with_subtitles, args=(input_path, output_path, uid, api_key)).start()

        flash('Video uploaded successfully! Processing with transcription and subtitles started.', 'success')
        return redirect(url_for('result', uid=uid))

    except Exception as e:
        print(f"[ERROR] Error processing upload: {e}")
        flash('Error processing video upload. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route("/result/<uid>")
def result(uid):
    """Show processing results and download links"""
    status = processing_status.get(uid, "unknown")
    
    # Check if files exist
    output_path = os.path.join(OUTPUT_DIR, f"output_{uid}.mp4")
    image_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumb_{uid}.jpg")
    video_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumbclip_{uid}.mp4")
    
    files_info = {
        'processed_video': os.path.exists(output_path),
        'image_thumbnail': os.path.exists(image_thumb_path),
        'video_thumbnail': os.path.exists(video_thumb_path),
        'uid': uid,
        'status': status
    }
    
    return render_template("result.html", **files_info)

@app.route("/status/<uid>")
def check_status(uid):
    """API endpoint to check processing status"""
    status = processing_status.get(uid, "unknown")
    output_path = os.path.join(OUTPUT_DIR, f"output_{uid}.mp4")
    is_ready = os.path.exists(output_path) and status == "completed"
    
    return jsonify({
        "status": status,
        "ready": is_ready
    })

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """Download processed video file"""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        flash('File not ready or not found', 'error')
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True)

@app.route("/download-subtitle/<uid>", methods=["GET"])
def download_subtitle(uid):
    """Download SRT subtitle file"""
    filename = f"subtitles_{uid}.srt"
    path = os.path.join(SUBTITLE_DIR, filename)
    if not os.path.exists(path):
        flash('Subtitle file not ready or not found', 'error')
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True, download_name=f"subtitles_{uid}.srt")

@app.route("/download-transcript/<uid>", methods=["GET"])
def download_transcript(uid):
    """Download raw transcript text file"""
    filename = f"transcript_{uid}.txt"
    path = os.path.join(TRANSCRIPT_DIR, filename)
    if not os.path.exists(path):
        flash('Transcript file not ready or not found', 'error')
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True, download_name=f"transcript_{uid}.txt")

@app.route("/thumbnail/<filename>", methods=["GET"])
def serve_thumbnail(filename):
    """Serve thumbnail files (image or video)"""
    path = os.path.join(THUMBNAIL_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Thumbnail not found"}), 404
    
    # Determine MIME type based on file extension
    if filename.lower().endswith('.mp4'):
        mimetype = "video/mp4"
    elif filename.lower().endswith(('.jpg', '.jpeg')):
        mimetype = "image/jpeg"
    elif filename.lower().endswith('.png'):
        mimetype = "image/png"
    else:
        mimetype = None
    
    return send_file(path, mimetype=mimetype)

@app.route("/api/process", methods=["POST"])
def api_process_video():
    """API endpoint for programmatic access"""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    # Get OpenAI API key from form data or JSON
    api_key = request.form.get('api_key') or request.json.get('api_key') if request.is_json else request.form.get('api_key')
    if not api_key:
        return jsonify({"error": "OpenAI API key is required"}), 400

    uid = uuid.uuid4().hex
    input_path = os.path.join(OUTPUT_DIR, f"input_{uid}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"output_{uid}.mp4")
    image_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumb_{uid}.jpg")
    video_thumb_path = os.path.join(THUMBNAIL_DIR, f"thumbclip_{uid}.mp4")

    try:
        file.save(input_path)
        print(f"[INFO] File saved to {input_path}")

        # Extract image thumbnail
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(image_thumb_path, frame)
                print(f"[INFO] Image thumbnail saved to {image_thumb_path}")
        cap.release()

        # Extract video thumbnail
        extract_video_thumbnail(input_path, video_thumb_path, start=5, duration=6)

        # Initialize processing status
        processing_status[uid] = "starting"

        # Start full processing with subtitles in background with API key
        threading.Thread(target=process_video_with_subtitles, args=(input_path, output_path, uid, api_key)).start()

        return jsonify({
            "message": "Processing with transcription and subtitles started",
            "uid": uid,
            "status_url": f"/status/{uid}",
            "video_download_url": f"/download/{os.path.basename(output_path)}",
            "thumbnail_image_url": f"/thumbnail/{os.path.basename(image_thumb_path)}",
            "thumbnail_video_url": f"/thumbnail/{os.path.basename(video_thumb_path)}",
            "subtitle_download_url": f"/download-subtitle/{uid}",
            "transcript_download_url": f"/download-transcript/{uid}"
        })

    except Exception as e:
        print(f"[ERROR] Error in API processing: {e}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
