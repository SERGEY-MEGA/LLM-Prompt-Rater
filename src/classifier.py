import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class PromptClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.safety_threshold = 0.35  # 🔒 КОНСЕРВАТИВНЫЙ ПОРОГ
    
    def fit(self, prompts, labels):
        X = self.embedder.encode(prompts, show_progress_bar=True)
        self.classifier.fit(X, labels)
        return self
    
    def predict(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        X = self.embedder.encode(prompts)
        proba = self.classifier.predict_proba(X)
        # 🔒 БЕЗОПАСНАЯ ЛОГИКА: если P(дизлайк) > 35% → блокируем
        preds = np.where(proba[:, 0] > self.safety_threshold, 0, 1)
        decisions = np.where(preds == 0, 'заблокировано', 'разрешено')
        return preds, proba, decisions
