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
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"모델 로딩 중: {model_id}...")
    
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa", 
    )

    # [수정됨] SFT / LoRA 수동 병합 및 화자 강제 등록 구간
    import os
    from safetensors.torch import load_file
    
    
    '''
    # 주의: 최신 학습이 완료된 에포크 폴더명으로 확인해서 맞춰주세요 (예: checkpoint-epoch-9)
    lora_dir = "./finetuning/output_sad_lora/checkpoint-epoch-9" 
    
    spk_embed_path = os.path.join(lora_dir, "target_speaker_embedding.pt")
    if os.path.exists(spk_embed_path):
        # 연구자님의 원본 코드대로, Qwen3TTSModel 안의 진짜 '.model' 껍데기를 한 겹 더 벗기고 들어갑니다.
        target_dtype = model.model.talker.model.codec_embedding.weight.dtype
        spk_embed = torch.load(spk_embed_path).to(model.device).to(target_dtype)
        # Base 모델의 빈 공간(3000번)에 0017번 성대 구조를 복사
        model.model.talker.model.codec_embedding.weight.data[ 3000 ] = spk_embed
        
        # 모델 설정(Config)이 0017번 화자를 인식하도록 맵핑 업데이트 (래퍼 내부의 진짜 모델 config로 한 겹 더 파고들어감)
        target_config = model.model.config
        talker_cfg = getattr(target_config, "talker_config", {})
        if isinstance(talker_cfg, dict):
            if "spk_id" not in talker_cfg: talker_cfg["spk_id"] = {}
            talker_cfg["spk_id"]["speaker_0017"] = 3000
        else:
            if getattr(talker_cfg, "spk_id", None) is None: talker_cfg.spk_id = {}
            talker_cfg.spk_id["speaker_0017"] = 3000
            
        # [추가됨] 에러를 피하기 위해 Base 모델의 이름표를 custom_voice로 강제 위조 (진짜 모델의 config만 수정)
        target_config.tts_model_type = "custom_voice"
        
        model.model.tts_model_type = "custom_voice"
        
        print("✅ 화자 지문 등록 완료: speaker_0017 (3000번)")
        
        
    # 2. Sad LoRA Task Vector 단일 병합 (직교 분리 주석 처리)
    emo_lora_path = os.path.join(lora_dir, "adapter_model.safetensors")
    
    if os.path.exists(emo_lora_path):
        print("🚀 Sad LoRA 가중치 단일 병합 시작...")
        emo_sd = load_file(emo_lora_path)
        # 껍데기가 아닌 진짜 모델(.model)의 가중치를 불러옵니다.
        base_sd = model.model.state_dict()
        
        # PEFT LoRA 가중치 이름(예: ...q_proj.lora_A.weight)에서 모듈 이름만 추출
        lora_modules = set()
        for k in emo_sd.keys():
            if "lora_A" in k:
                lora_modules.add(k.replace(".lora_A.weight", ""))
                
        scaling = 16 / 8  # lora_alpha / r (sft_12hz.py에서 설정한 값 기준)
        
        with torch.no_grad():
            for module_name in lora_modules:
                base_key = module_name.replace("base_model.model.", "") + ".weight"
                
                if base_key in base_sd:
                    base_w = base_sd[base_key]
                    
                    # 감정 LoRA의 Task Vector (delta_W) 복원: B @ A * scaling
                    emo_A = emo_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    emo_B = emo_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    delta_w_emo = (emo_B @ emo_A) * scaling
                    
                    # --- [추후 실험을 위한 직교(Orthogonal) 병합 로직 주석 보관] ---
                    # timbre_A = timbre_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    # timbre_B = timbre_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    # delta_w_timbre = (timbre_B @ timbre_A) * scaling
                    # flat_emo = delta_w_emo.flatten()
                    # flat_timbre = delta_w_timbre.flatten()
                    # dot_product = torch.dot(flat_timbre, flat_emo)
                    # norm_emo = torch.dot(flat_emo, flat_emo) + 1e-8
                    # projection = (dot_product / norm_emo) * delta_w_emo
                    # delta_w_timbre_orthogonal = delta_w_timbre - projection
                    # base_sd[base_key] = base_w + delta_w_emo.to(base_w.dtype) + delta_w_timbre_orthogonal.to(base_w.dtype)
                    # -------------------------------------------------------------
                    
                    # 현재는 순수하게 Sad 감정만 베이스에 덧셈 병합
                    base_sd[base_key] = base_w + delta_w_emo.to(base_w.dtype)
                    
        # 병합된 가중치를 껍데기가 아닌 진짜 모델(.model)에 다시 덮어씌웁니다.
        model.model.load_state_dict(base_sd)
        print("✅ Sad LoRA 단일 병합 완벽히 완료!")
    '''
    
    
    # 주의: 실제 파인튜닝하신 폴더 경로로 반드시 맞춰주십시오.
    sad_lora_dir = "./finetuning/output_sad_lora/checkpoint-epoch-9"
    neutral_lora_dir = "./finetuning/output_neutral_lora/checkpoint-epoch-9"
    
    # 1. 화자 지문 등록 (Sad와 Neutral 모두 동일한 화자이므로 Sad에서 추출)
    spk_embed_path = os.path.join(sad_lora_dir, "target_speaker_embedding.pt")
    if os.path.exists(spk_embed_path):
        target_dtype = model.model.talker.model.codec_embedding.weight.dtype
        spk_embed = torch.load(spk_embed_path).to(model.device).to(target_dtype)
        model.model.talker.model.codec_embedding.weight.data[ 3000 ] = spk_embed
        
        target_config = model.model.config
        talker_cfg = getattr(target_config, "talker_config", {})
        if isinstance(talker_cfg, dict):
            if "spk_id" not in talker_cfg: talker_cfg["spk_id"] = {}
            talker_cfg["spk_id"]["speaker_0017"] = 3000
        else:
            if getattr(talker_cfg, "spk_id", None) is None: talker_cfg.spk_id = {}
            talker_cfg.spk_id["speaker_0017"] = 3000
            
        target_config.tts_model_type = "custom_voice"
        model.model.tts_model_type = "custom_voice"
        print("✅ 화자 지문 등록 완료: speaker_0017 (3000번)")
        
    # 2. Task Vector Arithmetic (Sad - Neutral) 병합
    emo_lora_path = os.path.join(sad_lora_dir, "adapter_model.safetensors")
    neutral_lora_path = os.path.join(neutral_lora_dir, "adapter_model.safetensors")
    
    if os.path.exists(emo_lora_path) and os.path.exists(neutral_lora_path):
        print("🚀 Task Vector 빼기 연산(Sad - Neutral) 및 병합 시작...")
        emo_sd = load_file(emo_lora_path)
        neutral_sd = load_file(neutral_lora_path)
        base_sd = model.model.state_dict()
        
        lora_modules = set()
        for k in emo_sd.keys():
            if "lora_A" in k:
                lora_modules.add(k.replace(".lora_A.weight", ""))
                
        scaling = 16 / 8  # 기본 스케일링
        
        with torch.no_grad():
            for module_name in lora_modules:
                base_key = module_name.replace("base_model.model.", "") + ".weight"
                
                if base_key in base_sd:
                    base_w = base_sd[base_key]
                    
                    # Sad Task Vector 계산
                    emo_A = emo_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    emo_B = emo_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    delta_w_emo = (emo_B @ emo_A) * scaling
                    
                    # Neutral Task Vector 계산
                    neu_A = neutral_sd[f"{module_name}.lora_A.weight"].to(base_w.device).to(torch.float32)
                    neu_B = neutral_sd[f"{module_name}.lora_B.weight"].to(base_w.device).to(torch.float32)
                    delta_w_neu = (neu_B @ neu_A) * scaling
                    
                    # [핵심] 노이즈와 음색 교집합을 제거하고 순수 감정 벡터만 추출
                    delta_w_pure_sad = delta_w_emo - delta_w_neu
                    
                    # 추출된 순수 감정을 베이스 모델에 주입
                    # base_sd[base_key] = base_w + delta_w_pure_sad.to(base_w.dtype)
                    
                    # [추가됨] 뇌 회로가 붕괴되지 않도록, 정제된 '순수 감정' 성분에만 안전한 볼륨(1.5 ~ 2.0)을 곱해줍니다.
                    emotion_volume = 2.0
                    delta_w_pure_sad = delta_w_pure_sad * emotion_volume
                    
                    # 추출 및 증폭된 순수 감정을 베이스 모델에 주입
                    base_sd[base_key] = base_w + delta_w_pure_sad.to(base_w.dtype)
                    
        model.model.load_state_dict(base_sd)
        print("✅ 순수 슬픔 감정(Task Vector Arithmetic) 병합 완벽히 완료!")
        
        
        
    else:
        print("⚠️ 지정된 경로에 LoRA 파일이 없습니다. 원본 모델로 추론을 진행합니다.")
    

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
    
    # 4. 텍스트 설정 (통제 변인: 감정 유추가 절대 불가능한 건조하고 객관적인 정보 전달 문장)
    text = "The next train will arrive at the station at exactly 9 AM tomorrow morning."
    
    # 5. 실행 및 시간 측정
    print("음성 합성 진행 중...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        # [핵심] kwargs로 attn_type 전달
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="speaker_0017",
            instruct="",  # 모델의 자체 감정 생성 능력을 끄고, 오직 LoRA 가중치에만 의존하도록 강제함
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