import torch
import soundfile as sf
import os
import time
import csv
import argparse
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import load_file

# =====================================================================
# ⚙️ [실험 설정] 여기서 화자와 LoRA 경로를 조작하십시오.
# =====================================================================
# "aiden"을 입력하면 CustomVoice 모델로 구동되고, 
# "speaker_0017" 등을 입력하면 Base 모델로 자동 구동됩니다.
TARGET_SPEAKER = "speaker_0017" # "speaker_0017" "aiden"

EMOTION_LORA_DIR = "./finetuning/output_sad_lora/checkpoint-epoch-9"
NEUTRAL_LORA_DIR = "./finetuning/output_neutral_lora/checkpoint-epoch-9"
EMOTION_VOLUME = 2.0  # 순수 감정 증폭량 (기본 1.5 ~ 2.0 권장)
# =====================================================================

# --- 결과 폴더 이름 생성 함수 ---
def get_next_result_dir(base_name="results"):
    if not os.path.exists(base_name):
        return base_name
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
    parser = argparse.ArgumentParser(description="Qwen3-TTS Inference Experiment")
    parser.add_argument("--attn_type", type=str, default="None", 
                        choices=["None", "window_sdpa", "ours_sdpa", "h2o_sdpa"],
                        help="실험할 어텐션 타입 (None=순정, others_sdpa=커스텀)")
    args = parser.parse_args()

    current_attn_type = None if args.attn_type == "None" else args.attn_type
    print(f"\n🚀 실험 시작: Attn Type = {args.attn_type}, Target Speaker = {TARGET_SPEAKER}")

    # =====================================================================
    # 1. 동적 모델 로드 (화자에 따라 Base vs CustomVoice 자동 선택)
    # =====================================================================
    if TARGET_SPEAKER.lower() in ["aiden", "ryan"]:
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        print(f"✅ 내장 화자 감지 -> {model_id} 로딩 중...")
    else:
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        print(f"✅ 파인튜닝 화자({TARGET_SPEAKER}) 감지 -> {model_id} 로딩 중...")
    
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa", 
    )

    # =====================================================================
    # 2. 화자 지문 강제 등록 (Base 모델을 해킹하여 주입)
    # =====================================================================
    if TARGET_SPEAKER.lower() not in ["aiden", "ryan"]:
        target_config = model.model.config
        target_config.tts_model_type = "custom_voice"
        model.model.tts_model_type = "custom_voice"
        
        # [수정됨] 따로 추출해둔 speaker_embeddings 폴더에서 타겟 화자의 .pt 파일을 정확히 가져옵니다.
        spk_embed_path = os.path.join("./speaker_embeddings", f"{TARGET_SPEAKER}.pt")
        
        if os.path.exists(spk_embed_path):
            target_dtype = model.model.talker.model.codec_embedding.weight.dtype
            spk_embed = torch.load(spk_embed_path).to(model.device).to(target_dtype)
            
            # Base 모델의 빈 공간에 성대 복사
            model.model.talker.model.codec_embedding.weight.data[ 3000 ] = spk_embed
            
            talker_cfg = getattr(target_config, "talker_config", {})
            if isinstance(talker_cfg, dict):
                if "spk_id" not in talker_cfg: talker_cfg["spk_id"] = {}
                talker_cfg["spk_id"][TARGET_SPEAKER] = 3000
            else:
                if getattr(talker_cfg, "spk_id", None) is None: talker_cfg.spk_id = {}
                talker_cfg.spk_id[TARGET_SPEAKER] = 3000
                
            print(f"✅ Base 모델 해킹 완료: {TARGET_SPEAKER} 지문 등록 (3000번)")
    else:
        print("✅ CustomVoice 모델 순정 유지: 내장 화자 사용 준비 완료")

    # =====================================================================
    # 3. Task Vector Arithmetic (Emotion - Neutral) 병합
    # =====================================================================
    emotion_lora_path = os.path.join(EMOTION_LORA_DIR, "adapter_model.safetensors")
    neutral_lora_path = os.path.join(NEUTRAL_LORA_DIR, "adapter_model.safetensors")
    
    if os.path.exists(emotion_lora_path) and os.path.exists(neutral_lora_path):
        print(f"🚀 Task Vector 추출 및 직교 병합 시작... (볼륨: x{EMOTION_VOLUME})")
        emo_sd = load_file(emotion_lora_path)
        neutral_sd = load_file(neutral_lora_path)
        base_sd = model.model.state_dict()
        
        lora_modules = set()
        for k in emo_sd.keys():
            if "lora_A" in k:
                lora_modules.add(k.replace(".lora_A.weight", ""))
                
        scaling = 16 / 8  # lora_alpha / r
        
        with torch.no_grad():
            for module_name in lora_modules:
                base_key = module_name.replace("base_model.model.", "") + ".weight"
                
                if base_key in base_sd:
                    base_w = base_sd[base_key]
                    
                    # Emotion 벡터 계산
                    emo_A = emo_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    emo_B = emo_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    delta_w_emo = (emo_B @ emo_A) * scaling
                    
                    # Neutral 벡터 계산
                    neu_A = neutral_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    neu_B = neutral_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    delta_w_neu = (neu_B @ neu_A) * scaling
                    
                    # [핵심] 순수 감정 벡터 추출
                    delta_w_pure_emotion = delta_w_emo - delta_w_neu
                    
                    # 볼륨 증폭
                    delta_w_pure_emotion = delta_w_pure_emotion * EMOTION_VOLUME
                    
                    # 베이스 모델 가중치에 주입
                    base_sd[base_key] = base_w + delta_w_pure_emotion.to(base_w.dtype)
                    
        model.model.load_state_dict(base_sd)
        print("✅ 초순수 감정 가중치 주입 완벽히 완료!")
    else:
        print("⚠️ 경고: LoRA 파일을 찾을 수 없어 순정 상태로 진행합니다.")

    # =====================================================================
    # 4. 레이어 설정 및 카운터 초기화
    # =====================================================================
    layers = get_model_layers(model)
    if layers is None:
        print("⚠️ 경고: 모델 레이어를 찾을 수 없습니다.")
    else:
        print(f"✅ 모델 레이어 접근 성공 (총 {len(layers)}개)")
        for layer in layers:
            if hasattr(layer.self_attn, 'attn_type'): 
                layer.self_attn.attn_type = current_attn_type 
            if hasattr(layer.self_attn, 'decoding_cnt'):
                layer.self_attn.decoding_cnt = 0

    # =====================================================================
    # 5. 음성 추론 및 속도 측정
    # =====================================================================
    text = "The next train will arrive at the station at exactly 9 AM tomorrow morning."
    print("\n음성 합성 진행 중...")
    
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        # [핵심 방어막] 알리바바 모델의 깐깐한 화자 검증 로직 강제 무력화
        model._validate_speakers = lambda x: None
        
        # 모델 종류와 상관없이 이제 generate_custom_voice 단일 함수로 통일하여 사용
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker=TARGET_SPEAKER,  # 상단에서 지정한 화자 이름이 그대로 들어감
            instruct="",             # 모델의 자체 감정 해석 차단
            attn_type=current_attn_type 
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    total_duration = end_time - start_time

    # =====================================================================
    # 6. 결과 기록 및 오디오 저장
    # =====================================================================
    generated_tokens = 0
    if layers and len(layers) > 10 and hasattr(layers[10].self_attn, 'decoding_cnt'):
        generated_tokens = layers[10].self_attn.decoding_cnt
        print(f"📌 [Layer 10] 감지된 생성 토큰 수: {generated_tokens}")
    else:
        print("⚠️ decoding_cnt 미감지 -> Fallback 계산")
        audio_tensor = wavs[0] if isinstance(wavs, list) else wavs
        audio_len = audio_tensor.shape[-1] if hasattr(audio_tensor, 'shape') else len(audio_tensor)
        generated_tokens = (audio_len / sr) * 12 

    output_dir = get_next_result_dir("results")
    os.makedirs(output_dir, exist_ok=True)

    if generated_tokens > 0:
        tps = generated_tokens / total_duration
        ms_per_token = (total_duration / generated_tokens) * 1000
        
        print(f"\n[Performance Report - {args.attn_type} | Speaker: {TARGET_SPEAKER}]")
        print(f"✅ Total Time   : {total_duration:.4f} sec")
        print(f"✅ Total Tokens : {generated_tokens:.1f}")
        print(f"🚀 Speed        : {ms_per_token:.2f} ms/token ({tps:.2f} tokens/sec)")
        
        csv_file = os.path.join(output_dir, "speed_benchmark_qwen.csv")
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model", "Speaker", "Attn_Type", "Text_Len", "Total_Time", "Tokens", "ms_per_token"])
            writer.writerow(["Qwen3-TTS", TARGET_SPEAKER, args.attn_type, len(text), f"{total_duration:.4f}", generated_tokens, f"{ms_per_token:.2f}"])

    output_filename = "output_custom_voice.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    sf.write(output_path, wavs[0], sr)
    print(f"💾 저장 완료: {os.path.abspath(output_path)}\n")

if __name__ == '__main__':
    main()