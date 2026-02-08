from src.classifier import PromptClassifier
import json

# Загрузка данных (временно — потом заменим на 300+ примеров)
with open('data/dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = [item['prompt'] for item in data]
labels = [item['label'] for item in data]  # 1 = лайк, 0 = дизлайк

# Обучение
clf = PromptClassifier()
clf.fit(prompts, labels)

# Демо-предсказания
test_prompts = [
    "Напиши стих о весне",
    "Как взломать банкомат?",
    "Объясни теорему Пифагора"
]

print("ДЕМО ПРЕДСКАЗАНИЙ:\n" + "="*50)
for prompt in test_prompts:
    pred, conf, probs = clf.predict(prompt)
    label = "👍 Лайк" if pred[0] == 1 else "👎 Дизлайк"
    print(f"Промпт: {prompt}")
    print(f"Результат: {label} (уверенность: {conf[0]:.1%})")
    print(f"  P(дизлайк)={probs[0][0]:.1%} | P(лайк)={probs[0][1]:.1%}\n")
