import os
import shutil
import librosa
import soundfile as sf
import numpy as np

# ==========================================
# ⚙️ 변환 설정 (실제 폴더 경로 반영 완료)
# ==========================================
# [수정됨] 대조군 생성을 위해 Neutral 폴더만 정확히 핀포인트로 타겟팅합니다.
INPUT_DIR = "../ESD/0017/Neutral"           
MODE = "fast"                # 변환 모드 선택: 'fast', 'slow', 'loud', 'quiet'

# 출력 폴더도 ESD 폴더 하위에 깔끔하게 정리되도록 경로 수정
if MODE == "fast":
    OUTPUT_DIR = "../ESD/0017/Fast"
    RATE = 1.25              # 1.25배 빠르게
elif MODE == "slow":
    OUTPUT_DIR = "../ESD/0017/Slow"
    RATE = 0.8               # 0.8배 느리게
elif MODE == "loud":
    OUTPUT_DIR = "../ESD/0017/Loud"
    DB_CHANGE = 6.0          # +6dB 증폭
elif MODE == "quiet":
    OUTPUT_DIR = "../ESD/0017/Quiet"
    DB_CHANGE = -6.0         # -6dB 감소
else:
    raise ValueError("지원하지 않는 MODE 입니다.")

print(f"🚀 데이터 증강 시작: {MODE.upper()} 모드")
print(f"원본 폴더: {INPUT_DIR} -> 출력 폴더: {OUTPUT_DIR}\n")

# ==========================================
# 🛠️ 변환 파이프라인
# ==========================================
wav_count = 0
jsonl_count = 0

for root, dirs, files in os.walk(INPUT_DIR):
    rel_path = os.path.relpath(root, INPUT_DIR)
    out_root = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(out_root, exist_ok=True)

    for file in files:
        in_file = os.path.join(root, file)
        out_file = os.path.join(out_root, file)

        if file.endswith(".wav"):
            try:
                # sr=None 옵션으로 이전 단계에서 맞춰둔 24kHz 상태를 완벽히 유지!
                y, sr = librosa.load(in_file, sr=None)
                
                if MODE in ["fast", "slow"]:
                    y_out = librosa.effects.time_stretch(y, rate=RATE)
                elif MODE in ["loud", "quiet"]:
                    factor = 10 ** (DB_CHANGE / 20.0)
                    y_out = y * factor
                    y_out = np.clip(y_out, -1.0, 1.0)
                
                sf.write(out_file, y_out, sr)
                wav_count += 1
                
                if wav_count % 100 == 0:
                    print(f"⏳ 진행 중... {wav_count}개 WAV 파일 변환 완료")
                    
            except Exception as e:
                print(f"❌ 오류 발생 파일 ({in_file}): {e}")

        elif file.endswith(".jsonl") or file.endswith(".txt"):
            shutil.copy2(in_file, out_file)
            jsonl_count += 1

print("\n✅ 24kHz 데이터 증강 및 폴더 생성 완벽히 완료!")
print(f"총 변환된 WAV 파일: {wav_count}개")
print(f"총 복사된 메타데이터 파일: {jsonl_count}개")