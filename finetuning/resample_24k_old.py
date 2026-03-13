import os
import glob
import librosa
import soundfile as sf

def resample_all_audios(base_dir, target_sr=24000):
    # base_dir 하위의 모든 폴더를 뒤져서 wav 파일 경로를 싹 다 가져옵니다.
    wav_files = glob.glob(os.path.join(base_dir, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print(f"⚠️ {base_dir} 경로에서 wav 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return

    print(f"🚀 총 {len(wav_files)}개의 오디오 파일을 {target_sr}Hz 규격으로 변환 시작합니다...")
    
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
            
    print("✅ 모든 오디오 파일의 24kHz 변환이 완벽하게 완료되었습니다!")

if __name__ == '__main__':
    # 1. 0017번 화자의 슬픔(Sad) 데이터 폴더만 정확하게 핀포인트로 타겟팅
    sad_folder_path = os.path.abspath("../ESD/0017/Sad")                ##############
    resample_all_audios(sad_folder_path)
    
    # 2. 밖에 빼두었던 기준 파일(ref_sad.wav)도 단독으로 24kHz 변환 (필수)
    ref_file = os.path.abspath("../ESD/ref_sad.wav")                    ##############
    if os.path.exists(ref_file):
        import librosa, soundfile as sf
        audio, sr = librosa.load(ref_file, sr=24000)
        sf.write(ref_file, audio, 24000)
        print(f"✅ 기준 파일({ref_file}) 단독 24kHz 변환 완료!")