# train.py
import json
import numpy as np
from sklearn.model_selection import train_test_split
from src.classifier import PromptClassifier

# 1. Загрузка данных
print("Загрузка датасета...")
with open('data/dataset_v2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = [item['prompt'] for item in data]
labels = [item['label'] for item in data]

print(f"Всего примеров: {len(data)}")
print(f"Лайков (1): {sum(labels)} | Дизлайков (0): {len(labels) - sum(labels)}")

# 2. Разделение на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    prompts, labels, 
    test_size=0.2, 
    random_state=42,
    stratify=labels  # сохраняем баланс классов
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# 3. Обучение модели
print("\nОбучение модели...")
clf = PromptClassifier()
clf.fit(X_train, y_train)
print("✅ Модель обучена")

# 4. Оценка на тесте
print("\nОценка на тестовой выборке...")
preds, confidences, probs = clf.predict(X_test)

# Общая точность
accuracy = np.mean(np.array(preds) == np.array(y_test))
print(f"Общая точность: {accuracy:.2%}")

# Точность только на уверенных предсказаниях (≥70%)
high_conf_mask = confidences >= 0.7
high_conf_acc = np.mean(np.array(preds)[high_conf_mask] == np.array(y_test)[high_conf_mask])
high_conf_ratio = high_conf_mask.mean()

print(f"Уверенных предсказаний (≥70%): {high_conf_ratio:.1%}")
print(f"Точность на уверенных: {high_conf_acc:.2%}")

# 5. Анализ ошибок
print("\n❌ Примеры ошибок (топ-5 по низкой уверенности):")
errors = []
for i, (pred, true, conf, prompt) in enumerate(zip(preds, y_test, confidences, X_test)):
    if pred != true:
        errors.append((conf, prompt, pred, true))

errors_sorted = sorted(errors, key=lambda x: x[0])[:5]  # самые неуверенные ошибки
for conf, prompt, pred, true in errors_sorted:
    pred_label = "лайк" if pred == 1 else "дизлайк"
    true_label = "лайк" if true == 1 else "дизлайк"
    print(f"  Уверенность {conf:.1%} | '{prompt[:50]}...' → предсказано: {pred_label}, истина: {true_label}")

# 6. Сохранение модели (опционально)
import joblib
joblib.dump(clf, 'model.pkl')
print("\n✅ Модель сохранена в model.pkl")
