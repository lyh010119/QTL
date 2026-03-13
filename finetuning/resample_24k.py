import os
import glob
import librosa
import soundfile as sf

def resample_all_audios(base_dir, target_sr=24000):
    # base_dir 하위의 모든 폴더를 뒤져서 wav 파일 경로를 싹 다 가져옵니다.
    wav_files = glob.glob(os.path.join(base_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        return # 파일이 없으면 조용히 넘어갑니다.

    print(f"🚀 [{base_dir}] 폴더 내 {len(wav_files)}개 오디오 24kHz 변환 시작...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            # 1. librosa를 사용해 24kHz(target_sr)로 강제 변환하여 읽어 들입니다.
            audio, sr = librosa.load(wav_path, sr=target_sr)
            
            # 2. 변환된 오디오 데이터를 원본 경로에 그대로 덮어씌워 저장합니다.
            sf.write(wav_path, audio, target_sr)
            
            # 진행 상황 출력 (100개 단위)
            if (i + 1) % 100 == 0:
                print(f"   ... 진행 중: {i + 1} / {len(wav_files)} 완료")
                
        except Exception as e:
            print(f"❌ 에러 발생 ({wav_path}): {e}")
            
    print(f"✅ [{base_dir}] 폴더 변환 완료!\n")

if __name__ == '__main__':
    # 1. ESD 폴더의 절대 경로 설정
    esd_base_path = os.path.abspath("../ESD")
    
    # 2. 타겟팅할 감정 폴더 리스트
    target_emotions = ["Angry", "Sad", "Neutral"]
    
    print("=====================================================")
    print("🎙️ 다중 화자(0011~0020) 타겟 감정 오디오 24kHz 일괄 변환 시작")
    print("=====================================================\n")
    
    # 3. 0011 ~ 0020 화자 폴더 순회
    for folder_id in range(11, 21):
        folder_name = str(folder_id).zfill(4) # "0011" 형태로 맞춤
        
        # 각 화자 폴더 내의 특정 감정 폴더 순회
        for emotion in target_emotions:
            target_folder = os.path.join(esd_base_path, folder_name, emotion)
            
            if os.path.exists(target_folder):
                resample_all_audios(target_folder)
            else:
                print(f"⚠️ 폴더를 찾을 수 없어 건너뜁니다: {target_folder}")

    print("🎉 모든 타겟 화자 및 감정 폴더의 24kHz 변환이 완벽하게 끝났습니다!")