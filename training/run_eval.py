import os
from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import evaluate
import gc
import re
import pandas as pd
import csv
from src.data.dataset import EndoDataset
from src.models.endoreghd import EndoRegHD
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# [New user setup] Run via `bash scripts/run_eval.sh`, or `python
# training/run_eval.py` from the project root:
#   - DATA_DIR: official Kvasir-VQA-x1 test split (see data/, test-split
#     scripts: kvasir_download_test.py -> ... -> enrich_vqa_metadata_test.py).
#   - CHECKPOINT_PATH: the Stage 3 checkpoint to evaluate.
DATA_DIR = "kvasir_vqa_test"
JSON_PATH = os.path.join(DATA_DIR, "dataset.json")
LLM_PATH = "NousResearch/Llama-2-7b-chat-hf"

CHECKPOINT_PATH = "checkpoints/stage3_kvasir_v4/EndoRegHD_Epoch24.pth"
OUTPUT_DIR = "results/full_kvasir_eval_e24_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20
MAX_REGIONS = 6

def evaluate_model():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

    gc.collect()
    torch.cuda.empty_cache()

    print("Loading Metrics (ROUGE, METEOR, CHRF, BLEU, BERTScore)...")
    metrics = {
        "bleu": evaluate.load("bleu"),
        "rouge": evaluate.load("rouge"),
        "meteor": evaluate.load("meteor"),
        "chrf": evaluate.load("chrf"),
        "bertscore": evaluate.load("bertscore")
    }
    try:
        metrics["bleurt"] = evaluate.load("bleurt", "bleurt-20")
    except:
        print("WARNING: BLEURT checkpoint not found. Skipping BLEURT...")

    print("Initializing EndoRegHD with Stage 3.4 (SFT) Architecture...")
    model = EndoRegHD(llm_path=LLM_PATH, vision_ckpt_path=None, init_llm=False)

    base_llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_PATH,
        max_seq_length = 2048,
        load_in_4bit = True,
        trust_remote_code = True,
        dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if hasattr(model, 'llm'): del model.llm
    gc.collect()
    torch.cuda.empty_cache()
    
    model.llm = FastLanguageModel.get_peft_model(
        base_llm,
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Stage 3.4 weights: {CHECKPOINT_PATH}")
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = sd['model_state_dict'] if 'model_state_dict' in sd else sd
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            if not any(ext in k for ext in [".absmax", ".quant_map", ".nested_", ".quant_state"]):
                new_key = k.replace("_orig_mod.", "")
                clean_state_dict[new_key] = v

        model.load_state_dict(clean_state_dict, strict=False)
        print("Successfully matched and loaded the original weights!")
    else:
        print("Critical: Checkpoint NOT FOUND!")
        return

    if hasattr(model.llm, "config"):
        model.llm.config.vocab_size = 32000
        model.llm.generation_config.max_length = None

    print("Enabling PyTorch 2.0 Compile for the inference graph...")
    model = torch.compile(model)

    for name, m in model.named_children():
        if name != "llm": m.to(DEVICE)
    model.eval()

    test_dataset = EndoDataset(DATA_DIR, JSON_PATH, tokenizer, mode='val', max_regions=MAX_REGIONS)
    
    def collate_eval(batch):
        return torch.stack([item['global'] for item in batch]), \
               torch.stack([item['texture'] for item in batch]), \
               torch.stack([item['geometry'] for item in batch]), \
               [{'question': item['question'], 'answer': item['answer'], 'num_regions': item['num_regions']} for item in batch]

    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval)

    predictions = []
    references = []

    csv_path = os.path.join(OUTPUT_DIR, "predictions_debug.csv")
    with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Reference", "Prediction"])

    print(f"Starting Inference on {len(test_dataset)} samples with BATCH_SIZE={BATCH_SIZE}...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            g_img, t_img, geo_img, meta = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE), batch[3]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                g_emb, r_emb = model.forward_vision_features(g_img, t_img, geo_img)
                prompt_embeds, _, attention_mask = model.prepare_multimodal_prompt(
                    g_emb, r_emb, meta, tokenizer, DEVICE, 
                    inference=True
                )
                
                B, SeqLen, D = prompt_embeds.shape
                fixed_embeds = torch.zeros_like(prompt_embeds)
                fixed_mask = torch.zeros_like(attention_mask)

                for b in range(B):
                    valid_idx = attention_mask[b].nonzero(as_tuple=True)[0]
                    num_valid = len(valid_idx)
                    num_pad = SeqLen - num_valid

                    fixed_embeds[b, num_pad:] = prompt_embeds[b, valid_idx]
                    fixed_mask[b, num_pad:] = attention_mask[b, valid_idx]

                prompt_embeds = fixed_embeds
                attention_mask = fixed_mask
                
                out_ids = model.llm.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False, 
                    num_beams=1,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            with open(csv_path, mode="a", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                for i in range(len(meta)):
                    decoded_text = tokenizer.decode(out_ids[i], skip_special_tokens=True).strip()
                    final_pred = decoded_text.split("Answer:")[-1].strip() if "Answer:" in decoded_text else decoded_text
                    
                    predictions.append(final_pred)
                    references.append(meta[i]['answer'])
                    
                    writer.writerow([meta[i]['answer'], final_pred])

    print(f"\nFinished inference on {len(predictions)} samples and saved realtime data successfully!")

    print("\nStarting metrics computation...")
    results = {}

    predictions_lower = [p.lower() for p in predictions]
    references_lower = [r.lower() for r in references]

    for n in [1, 2, 4]:
        print(f"Computing BLEU-{n}...")
        b = metrics["bleu"].compute(predictions=predictions_lower, references=references_lower, max_order=n)
        results[f"BLEU-{n}"] = b['bleu'] * 100

    print("Computing ROUGE (1, 2, L)...")
    r = metrics["rouge"].compute(predictions=predictions_lower, references=references_lower, use_stemmer=True)
    results["ROUGE-1"] = r["rouge1"] * 100
    results["ROUGE-2"] = r["rouge2"] * 100
    results["ROUGE-L"] = r["rougeL"] * 100

    print("Computing METEOR...")
    results["METEOR"] = metrics["meteor"].compute(predictions=predictions_lower, references=references_lower)["meteor"] * 100

    print("Computing CHRF++...")
    results["CHRF++"] = metrics["chrf"].compute(predictions=predictions_lower, references=references_lower)["score"]

    print("Computing BERTScore (using RoBERTa-large)...")
    try:
        bs = metrics["bertscore"].compute(
            predictions=predictions_lower,
            references=references_lower,
            lang="en",
            model_type="roberta-large",
            batch_size=32
        )
        results["BERT-F1"] = sum(bs["f1"]) / len(bs["f1"]) * 100
        print("BERTScore computed successfully!")
    except Exception as e:
        print(f"WARNING: BERTScore failed (possibly due to running out of RAM): {e}")
        results["BERT-F1"] = 0.0

    results["BLEURT"] = 0.0

    df_metrics = pd.DataFrame([results])
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)

    print("\n" + "=" * 20 + " FINAL RESULTS " + "=" * 20)
    for k, v in results.items():
        print(f" {k:12}: {v:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    evaluate_model()