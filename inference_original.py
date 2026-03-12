import torch
import soundfile as sf
import os
import time
import csv
import argparse
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# --- 결과 폴더 이름 생성 함수 ---
def get_next_result_dir(base_name="results"):
    """폴더가 존재하면 results(1), results(2) 형식으로 이름을 반환합니다."""
    # base_name 자체가 없으면 바로 사용
    if not os.path.exists(base_name):
        return base_name
    
    # results(1), results(2)... 순으로 찾기
    count = 1
    while True:
        new_name = f"{base_name}({count})"
        if not os.path.exists(new_name):
            return new_name
        count += 1

# --- 모델 내부 레이어 접근 함수 ---
def get_model_layers(wrapper_model):
    try:
        return wrapper_model.model.talker.model.layers
    except AttributeError:
        for name, module in wrapper_model.named_modules():
            if hasattr(module, 'layers') and isinstance(module.layers, torch.nn.ModuleList):
                return module.layers
    return None

def main():
    # 1. 인자 파싱 (run.sh에서 받아옴)
    parser = argparse.ArgumentParser(description="Qwen3-TTS Inference Experiment")
    parser.add_argument("--attn_type", type=str, default="None", 
                        choices=["None", "window_sdpa", "ours_sdpa", "h2o_sdpa"],
                        help="실험할 어텐션 타입 (None=순정, others_sdpa=커스텀)")
    args = parser.parse_args()

    # attn_type 문자열 처리 ("None" 문자열이 들어오면 파이썬 None으로 변환)
    current_attn_type = None if args.attn_type == "None" else args.attn_type
    
    print(f"\n🚀 실험 시작: Attn Type = {args.attn_type}")

    # 2. 모델 로드
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    print(f"모델 로딩 중: {model_id}...")
    
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa", 
    )

    # 3. 레이어 설정 및 카운터 초기화
    layers = get_model_layers(model)
    if layers is None:
        print("⚠️ 경고: 모델 레이어를 찾을 수 없습니다.")
    else:
        print(f"✅ 모델 레이어 접근 성공 (총 {len(layers)}개)")
        for layer in layers:
            # [핵심] 외부에서 받은 attn_type을 각 레이어에 주입 (modeling 코드에서 self.attn_type을 참조하거나 kwargs로 받게 됨)
            if hasattr(layer.self_attn, 'attn_type'): 
                layer.self_attn.attn_type = current_attn_type # 속성 주입 (보조 수단)
            
            # 카운터 초기화
            if hasattr(layer.self_attn, 'decoding_cnt'):
                layer.self_attn.decoding_cnt = 0

    # 4. 텍스트 설정
    text = "현대 과학 기술의 정점이라 불리는 인공지능 기반의 음성 합성 시스템은 단순히 텍스트를 소리로 변환하는 단계를 넘어 인간이 가진 고유의 감정과 억양 그리고 문맥 속에 숨겨진 미묘한 뉘앙스까지 완벽하게 재현하려는 원대한 목표를 가지고 끊임없이 발전하고 있다."
    
    # 5. 실행 및 시간 측정
    print("음성 합성 진행 중...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        # [핵심] kwargs로 attn_type 전달 (modeling에서 kwargs.get("attn_type")으로 받음)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Korean",
            speaker="Sohee",
            instruct="차분하고 명확한 목소리로",
            attn_type=current_attn_type  # 여기서 전달!
        )

    torch.cuda.synchronize()
    end_time = time.time()
    total_duration = end_time - start_time

    # 6. 속도 및 토큰 계산
    generated_tokens = 0
    if layers and len(layers) > 10 and hasattr(layers[10].self_attn, 'decoding_cnt'):
        generated_tokens = layers[10].self_attn.decoding_cnt
        print(f"📌 [Layer 10] 감지된 생성 토큰 수: {generated_tokens}")
    else:
        # Fallback (12Hz 모델 기준)
        print("⚠️ decoding_cnt 미감지 -> Fallback 계산")
        audio_tensor = wavs[0] if isinstance(wavs, list) else wavs
        audio_len = audio_tensor.shape[-1] if hasattr(audio_tensor, 'shape') else len(audio_tensor)
        generated_tokens = (audio_len / sr) * 12 

    # --- 결과 폴더 먼저 생성 ---
    output_dir = get_next_result_dir("results") # results, results(1), results(2)...
    os.makedirs(output_dir, exist_ok=True)

    # 7. 결과 출력 및 저장
    if generated_tokens > 0:
        tps = generated_tokens / total_duration
        ms_per_token = (total_duration / generated_tokens) * 1000
        
        print(f"\n[Performance Report - {args.attn_type}]")
        print(f"✅ Total Time   : {total_duration:.4f} sec")
        print(f"✅ Total Tokens : {generated_tokens:.1f}")
        print(f"🚀 Speed        : {ms_per_token:.2f} ms/token ({tps:.2f} tokens/sec)")
        
        csv_file = os.path.join(output_dir, "speed_benchmark_qwen.csv")
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model", "Attn_Type", "Text_Len", "Total_Time", "Tokens", "ms_per_token"])
            writer.writerow(["Qwen3-TTS", args.attn_type, len(text), f"{total_duration:.4f}", generated_tokens, f"{ms_per_token:.2f}"])

    # 8. 오디오 파일 저장
    # 파일명에 attn_type 포함 (주석대로 output_custom_voice.wav 사용)
    # output_filename = f"output_{args.attn_type}.wav"
    output_filename = f"output_custom_voice.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    sf.write(output_path, wavs[0], sr)
    print(f"💾 저장 완료: {os.path.abspath(output_path)}\n")

if __name__ == '__main__':
    main()