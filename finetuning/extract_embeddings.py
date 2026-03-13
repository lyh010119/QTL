import torch
import json
import os
import glob
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.utils.data import DataLoader

# =========================================================
# ⚙️ 설정: 연구자님이 갖고 계신 수백 줄짜리 전체 Neutral JSONL을 그대로 넣으십시오.
# (스크립트가 알아서 화자 이름별로 딱 1개만 뽑고 나머지는 똑똑하게 스킵합니다.)
# =========================================================
REF_JSONL_PATH = "../ESD/train_neutral_with_codes.jsonl"
OUTPUT_DIR = "./speaker_embeddings"

def main():
    print("🚀 Base 모델 로딩 중 (지문 추출기용)...")
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # JSONL 읽기
    lines = open(REF_JSONL_PATH, "r", encoding="utf-8").readlines()
    data = [json.loads(line) for line in lines]
    
    # Dataset 객체 생성 (전처리를 위해 필요)
    print("⏳ 데이터셋 전처리 중...")
    dataset = TTSDataset(data, qwen3tts.processor, qwen3tts.model.config)
    
    print(f"✅ 총 {len(dataset)}개의 문장 확인 완료. 중복을 제외하고 화자별 지문 추출을 시작합니다...\n")
    
    # 중복 추출 방지용 세트(Set)
    extracted_speakers = set()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # JSONL에서 오디오 경로를 읽어와서 폴더 이름(예: "0011")을 화자 이름으로 추출합니다.
            # 경로 예시: /home/yhlee/.../ESD/0011/Neutral/0011_000001.wav
            wav_path = data[i].get("audio", "")
            if not wav_path:
                continue
                
            # 경로에서 화자 ID(0011 등) 파싱
            path_parts = wav_path.split(os.sep)
            try:
                # "Neutral" 폴더 바로 상위 폴더 이름이 화자 번호입니다.
                neutral_idx = path_parts.index("Neutral")
                speaker_id = path_parts[neutral_idx - 1] 
                spk_name = f"speaker_{speaker_id}" # "speaker_0011" 형태 완성
            except ValueError:
                spk_name = f"speaker_unknown_{i}"
            
            # 이미 추출한 화자면 쿨하게 스킵! (시간 절약)
            if spk_name in extracted_speakers:
                continue
                
            # ----------------------------------------------------
            # 🎙️ 지문(Embedding) 추출 핵심 로직
            # ----------------------------------------------------
            item = dataset[i]
            # [핵심 수술] collate_fn을 통과시켜야 오디오가 멜스펙트로그램으로 완벽히 변환된 '배치(Batch)' 형태가 됩니다!
            batch = dataset.collate_fn([item]) 
            
            # collate_fn이 이미 배치 차원(1)을 만들어주었으므로 unsqueeze(0)은 뺍니다.
            ref_mels = batch['ref_mels'].to(qwen3tts.device).to(torch.bfloat16)
            
            
            # Base 모델의 성대 분석기를 통과시켜 지문 벡터 도출
            speaker_embedding = qwen3tts.model.speaker_encoder(ref_mels).detach().cpu()
            
            # .pt 파일로 저장
            pt_path = os.path.join(OUTPUT_DIR, f"{spk_name}.pt")
            torch.save(speaker_embedding[0], pt_path)
            
            extracted_speakers.add(spk_name)
            print(f"🎉 지문 추출 완료: {spk_name} -> {pt_path}")

    print("\n✅ 모든 화자의 지문 추출이 완벽하게 끝났습니다! 이제 추론 코드를 돌릴 수 있습니다.")

if __name__ == "__main__":
    main()