from fastapi import FastAPI
from video_processor import VideoProcessor
import uvicorn

app = FastAPI()

@app.post('/analyze_video')
async def analyze_video(filename: str):
    path = f"../uploads/{filename}"
    processor = VideoProcessor()
    results = processor.process_video_file(path)
    return {"video": filename, "results": results}

if __name__ == '__main__':
    uvicorn.run("app:app", reload=True, port=8000)