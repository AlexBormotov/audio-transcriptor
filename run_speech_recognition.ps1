param(
    [Parameter(Position = 0)]
    [string] $MediaFile
)

# File: python transcriber.py <path> — same pipeline as GUI (ffmpeg + Whisper).
# No argument: legacy mode — microphone + speech_recognition_online.py (requires CUDA).
if ($MediaFile) {
    Write-Host "Transcribing file: $MediaFile" -ForegroundColor Green
    python "$PSScriptRoot\transcriber.py" --model base "$MediaFile"
} else {
    Write-Host "Microphone (speech_recognition_online.py)..." -ForegroundColor Green
    python "$PSScriptRoot\speech_recognition_online.py"
}
Write-Host "Done." -ForegroundColor Yellow
Pause
