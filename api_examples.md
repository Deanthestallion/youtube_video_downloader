# Video Processing API - HTTP Usage Examples

## Web Form Upload
The simplest way is through the web interface at your app URL. Just provide:
1. Your OpenAI API key (starts with `sk-`)
2. Select your video file
3. Click "Process Video"

## API Endpoint Usage

### 1. Upload Video with curl
```bash
# Basic upload with API key
curl -X POST "https://your-replit-app.replit.app/api/process" \
  -F "file=@your_video.mp4" \
  -F "api_key=sk-your-actual-openai-api-key-here"
```

### 2. Check Processing Status
```bash
# Replace {uid} with the unique ID returned from upload
curl "https://your-replit-app.replit.app/status/{uid}"

# Example response:
# {"status": "transcribing", "ready": false}
# {"status": "completed", "ready": true}
```

### 3. Download Processed Video
```bash
# Download the processed video with subtitles burned in
curl "https://your-replit-app.replit.app/download/output_{uid}.mp4" \
  -o processed_video.mp4
```

### 4. Download Subtitle File
```bash
# Download the English SRT subtitle file
curl "https://your-replit-app.replit.app/download-subtitle/{uid}" \
  -o subtitles.srt
```

## JavaScript Example
```javascript
const formData = new FormData();
formData.append('file', videoFile); // File object from input
formData.append('api_key', 'sk-your-openai-api-key');

fetch('/api/process', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Upload successful:', data);
    // Use data.uid to check status and download files
});
```

## Python Example
```python
import requests

# Upload video
files = {'file': open('video.mp4', 'rb')}
data = {'api_key': 'sk-your-openai-api-key'}

response = requests.post('https://your-app.replit.app/api/process', 
                        files=files, data=data)
result = response.json()
uid = result['uid']

# Check status
status_response = requests.get(f'https://your-app.replit.app/status/{uid}')
print(status_response.json())

# Download when ready
if status_response.json()['ready']:
    video_response = requests.get(f'https://your-app.replit.app/download/output_{uid}.mp4')
    with open('processed_video.mp4', 'wb') as f:
        f.write(video_response.content)
```

## Complete Workflow Example
```bash
#!/bin/bash
# Complete video processing workflow

API_KEY="sk-your-openai-api-key"
VIDEO_FILE="input_video.mp4"
BASE_URL="https://your-app.replit.app"

echo "Uploading video..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/process" \
  -F "file=@$VIDEO_FILE" \
  -F "api_key=$API_KEY")

UID=$(echo $RESPONSE | grep -o '"uid":"[^"]*"' | cut -d'"' -f4)
echo "Processing started with UID: $UID"

# Wait for completion
while true; do
    STATUS=$(curl -s "$BASE_URL/status/$UID" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    echo "Status: $STATUS"
    
    if [ "$STATUS" = "completed" ]; then
        echo "Processing completed!"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "Processing failed!"
        exit 1
    fi
    
    sleep 10
done

# Download results
echo "Downloading processed video..."
curl "$BASE_URL/download/output_$UID.mp4" -o "processed_$VIDEO_FILE"

echo "Downloading subtitles..."
curl "$BASE_URL/download-subtitle/$UID" -o "subtitles_$UID.srt"

echo "Downloading transcript..."
curl "$BASE_URL/download-transcript/$UID" -o "transcript_$UID.txt"

echo "Done! Files saved as processed_$VIDEO_FILE, subtitles_$UID.srt, and transcript_$UID.txt"
```

## Response Format
When you upload a video, you'll get a JSON response like:
```json
{
  "message": "Processing with transcription and subtitles started",
  "uid": "1234567890abcdef",
  "status_url": "/status/1234567890abcdef",
  "video_download_url": "/download/output_1234567890abcdef.mp4",
  "subtitle_download_url": "/download-subtitle/1234567890abcdef",
  "thumbnail_image_url": "/thumbnail/thumb_1234567890abcdef.jpg",
  "thumbnail_video_url": "/thumbnail/thumbclip_1234567890abcdef.mp4"
}
```

## Processing Status Values
- `starting` - Upload received, initializing
- `removing_silence` - Auto-editor removing silent parts
- `detecting_faces` - AI detecting faces for zoom
- `extracting_audio` - Extracting audio for transcription
- `transcribing` - AI detecting language and transcribing
- `creating_subtitles` - Translating and creating English subtitles
- `burning_subtitles` - Burning subtitles into video
- `completed` - All done, files ready for download
- `failed` - Something went wrong

## Important Notes
- Replace `your-replit-app.replit.app` with your actual Replit app URL
- Your OpenAI API key must start with `sk-` and have valid permissions
- Supported video formats: MP4, AVI, MOV, MKV, WebM
- The app automatically detects any language and translates to English
- Processing time depends on video length (typically 2-5 minutes for short videos)