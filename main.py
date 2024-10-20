import os
import gc
import numpy as np
import cv2
import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import pytesseract
from PIL import Image
from pydub import AudioSegment
import librosa
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import tensorflow as tf
from typing import List, Tuple

# 터미널에 uvicorn main:app --reload 로 실행
app = FastAPI()

# Keras 모델 로드
model = tf.keras.models.load_model("241013 이진분류_attention+평가정확도.keras")

# 정적 파일 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# Tesseract OCR 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

product_index = {
    "리디잇": "의약외품",
    "리디핏": "의약외품",
    "닥터33": "탈모 관련",
    "닥터두피톡스": "탈모 관련",
    "닥터란": "탈모 관련",
    "닥터모리엔": "탈모 관련",
    "미노샷": "탈모 관련",
    "닥터모나": "탈모 관련",
    "2BR": "탈모 관련",
    "씨커트": "건강기능식품",
    "본투식스": "건강기능식품",
    "뉴티락": "건강기능식품",
    "뉴티엠": "건강기능식품",
    "오라컷플러스": "건강기능식품",
    "365컷": "건강기능식품",
    "리디샷": "건강기능식품",
    "리디혈": "건강기능식품",
    "판토엔": "건강기능식품",
    "요오드밸런스포우먼": "건강기능식품",
    "조인트큐어포우먼": "건강기능식품",
    "투비컷": "건강기능식품",
    "라이트유산균포우먼": "건강기능식품",
    "간편엔": "건강기능식품",
    "눈편엔": "건강기능식품",
    "위편엔": "건강기능식품",
    "브이젠": "건강기능식품",
    "셀렌톡": "건강기능식품",
    "혈행뻥": "건강기능식품",
    "스카이메모리": "건강기능식품",
    "바디미스트": "화장품",
    "샤비크": "화장품",
    "실프팅앰플": "화장품",
    "칼슘아이크림": "화장품",
    "스피큐락셀": "화장품",
    "스피큐락셀앰플": "화장품",
    "리턴엔": "화장품",
    "톤업엔": "화장품",
    "반달크림": "화장품",
    "에이트크림": "화장품",
    "핸드문크림": "화장품",
    "포레티": "화장품",
    "아크밀리": "화장품",
    "주미멀티밤": "화장품",
    "컬리랩": "화장품",
    "다트너스20.9": "다이어트 관련",
    "제로픽": "다이어트 관련",
    "디커트프리미엄": "다이어트 관련",
    "제로픽플러스": "다이어트 관련",
    "디다샷": "다이어트 관련",
    "와이블라썸": "여성용품",
    "와이이뮤": "여성용품",
    "아이바른풋": "아이, 청소년용",
    "아이뽀밤부": "아이, 청소년용",
    "아이치카푸": "아이, 청소년용",
    "아이튼튼츄": "아이, 청소년용",
    "연료첨가제": "자동차용품",
    "유리막코팅제": "자동차용품",
    "흠집제거제": "자동차용품"
}

category_index = {
    '건강기능식품': 0, 
    '다이어트 관련': 1, 
    '아이, 청소년용': 2, 
    '여성용품': 3, 
    '의약외품': 4, 
    '자동차용품': 5, 
    '탈모 관련': 6, 
    '화장품': 7
}

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b!=0)

# 영상데이터 추출
def extract_features(video_path, n_samples=50):
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = convnext_base(weights=weights)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames >= n_samples:
        indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames + [frames[-1]] * (n_samples - total_frames)

    features = []
    with torch.no_grad():
        for frame in sampled_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            frame = frame.unsqueeze(0)
            frame = frame / 255.0
            frame = ConvNeXt_Base_Weights.DEFAULT.transforms()(frame)
            feature = model.features(frame)
            feature = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1))
            features.append(feature.squeeze().numpy())

    return np.array(features)

# 프레임 전처리
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# 텍스트 검출
def detect_text(frame):
    preprocessed = preprocess_frame(frame)
    text = pytesseract.image_to_data(Image.fromarray(preprocessed), output_type=pytesseract.Output.DICT)
    total_text_area = sum(int(text['width'][i]) * int(text['height'][i]) for i in range(len(text['text'])) if len(text['text'][i]) > 0)
    return total_text_area

# 프레임 분석
def analyze_frame(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    text_area = detect_text(frame)
    frame_area = frame.shape[0] * frame.shape[1]
    text_ratio = text_area / frame_area
    
    return len(faces), text_ratio

# audio 특성 추출
def extract_audio_features(video_path):
    try:
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export('temp.wav', format="wav")
        
        y, sr = librosa.load('temp.wav', duration=60)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate),
        }
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        features = {}
        
    finally:
        if os.path.exists('temp.wav'):
            os.remove('temp.wav')
    
    return features

def analyze_video(video_path, sample_interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(int(fps * 60), total_frames)
    
    face_frames = 0
    total_text_ratio = 0
    
    for frame_idx in range(0, frames_to_process, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            faces, text_ratio = analyze_frame(frame)
            if faces > 0:
                face_frames += 1
            total_text_ratio += text_ratio
        except Exception:
            continue
    
    cap.release()
    
    frames_processed = frames_to_process // sample_interval
    if frames_processed == 0:
        raise ValueError("No frames were successfully processed")

    results = {
        'face_ratio': safe_divide(face_frames, frames_processed),
        'avg_text_ratio': safe_divide(total_text_ratio, frames_processed),
    }
    
    try:
        audio_features = extract_audio_features(video_path)
        results.update(audio_features)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
    
    return results, total_frames

async def process_video(file: UploadFile) -> Tuple[np.ndarray, np.ndarray]:
    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # video_features 추출 (50, 1024) 구조 유지
        video_features = extract_features(temp_path)
        
        # audio_features 추출
        video_results, _ = analyze_video(temp_path)
        
        # 오디오 특성 분리 및 (54,) 구조로 변환
        audio_features = []     
        
        for _, value in video_results.items():
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    audio_features.append(float(value))
                else:
                    audio_features.extend(value.flatten().tolist())
            elif isinstance(value, (int, float, np.number)):
                audio_features.append(float(value))
            elif isinstance(value, list):
                audio_features.extend(value)
        
        audio_features = np.array(audio_features)
        
        return video_features, audio_features
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "products": list(product_index.keys())})

@app.post("/predict/")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    product: str = Form(...),
    size: int = Form(...)
):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "products": list(product_index.keys()),
            "error": "Invalid file format. Please upload a video file."
        })

    try:
        video_features, audio_features = await process_video(file)
        
        category = product_index.get(product, "Unknown")
        category_label = category_index.get(category, -1)
        
        # 모델 입력 형식 조정
        category_label = np.array([category_label])
        size = np.array([size])
        
        # 디버깅: 각 입력의 shape 출력
        print(f"video_features shape: {video_features.shape}")
        print(f"audio_features shape: {audio_features.shape}")
        print(f"category_label shape: {category_label.shape}")
        print(f"size shape: {size.shape}")
        
        # 모델 입력을 단일 배치로 구성
        model_input = [
            video_features.reshape(1, *video_features.shape),
            audio_features.reshape(1, *audio_features.shape),
            category_label.reshape(1, -1),
            size.reshape(1, -1)
        ]
        
        prediction = model.predict(model_input)

        print(prediction)
        
        result = "승인" if prediction[0][0] > 0.5 else "거부"
        confidence = float(prediction[0][0])

        return templates.TemplateResponse("result.html", {
            "request": request,
            "prediction": result,
            "confidence": confidence,
            "video_features_length": video_features.shape[0],
            "audio_features_length": audio_features.shape[0],
            "product": product,
            "category": category,
            "category_label": int(category_label[0]),
            "size": int(size[0])
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "products": list(product_index.keys()),
            "error": f"An error occurred during prediction: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)