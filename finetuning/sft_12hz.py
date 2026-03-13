# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16", log_with="tensorboard", project_dir=args.output_model_path) # 원래는 4
    MODEL_PATH = args.init_model_path

    '''
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    '''
    
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # --- [추가됨] PEFT(LoRA) 설정 ---
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=32, # 8
        lora_alpha=64, # 16
        target_modules="all-linear", # 모델 내부의 모든 레이어에 LoRA를 안전하게 부착
        bias="none",
    )
    qwen3tts.model = get_peft_model(qwen3tts.model, lora_config)
    qwen3tts.model.print_trainable_parameters() # 터미널에 줄어든 파라미터 비율 출력
    # ---------------------------------

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # [수정됨] 전체 모델 파라미터가 아닌, LoRA로 인해 requires_grad=True가 된 가벼운 파라미터만 학습하도록 필터링
    optimizer = AdamW(filter(lambda p: p.requires_grad, qwen3tts.model.parameters()), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")


        '''
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            
            # [복구 및 개선됨] 파이토치 모델에서 직접 config를 추출하여 custom_voice 모드로 갱신 후 안전하게 저장
            import json
            config_dict = model.config.to_dict()
            config_dict["tts_model_type"] = "custom_voice"
            
            talker_config = config_dict.get("talker_config", {})
            if "spk_id" not in talker_config:
                talker_config["spk_id"] = {}
            talker_config["spk_id"][args.speaker_name] = 3000
            
            if "spk_is_dialect" not in talker_config:
                talker_config["spk_is_dialect"] = {}
            talker_config["spk_is_dialect"][args.speaker_name] = False
            
            config_dict["talker_config"] = talker_config
            
            output_config_file = os.path.join(output_dir, "config.json")
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        '''      
        
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            
            # [수정됨] 허깅페이스 모델 ID를 로컬 폴더로 착각하여 복사하려는 오류(shutil.copytree) 방지.
            # LoRA 파인튜닝이므로 원본 모델과 config를 통째로 복사할 필요 없이 안전하게 저장할 빈 폴더만 생성합니다.        
                
                
            unwrapped_model = accelerator.unwrap_model(model)
            
            # [수정됨] 1.7B 전체 모델을 무식하게 저장하는 대신, peft 라이브러리를 통해 가벼운 LoRA 어댑터(수십 MB)만 깔끔하게 저장
            unwrapped_model.save_pretrained(output_dir)
            
            # (옵션) 원본 코드에서 특정 스피커 임베딩을 저장하려던 로직 분리. 추론(Inference) 시 활용하기 위해 별도 텐서로 저장해둠
            if target_speaker_embedding is not None:
                torch.save(target_speaker_embedding[0].detach().cpu(), os.path.join(output_dir, "target_speaker_embedding.pt"))

if __name__ == "__main__":
    train()
