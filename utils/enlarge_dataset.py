from datasets import load_dataset

# 1. Jailbreak-промпты (дизлайки)
jb = load_dataset("JailbreakBench/JailbreakBench", split="train")
bad_prompts = [ex['prompt'] for ex in jb if ex['attack_type'] != "benign"][:120]

# 2. ToxicChat (дизлайки)
toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
bad_prompts += [ex['user_input'] for ex in toxic if ex['toxicity'] == 1][:80]

# 3. Сохрани как часть датасета
import json
with open('data/bad_prompts_raw.json', 'w', encoding='utf-8') as f:
    json.dump(bad_prompts, f, ensure_ascii=False, indent=2)

print(f"Скачано {len(bad_prompts)} вредоносных промптов")
