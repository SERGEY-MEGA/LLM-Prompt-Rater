import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class PromptClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # боремся с дисбалансом
        )
        self.threshold = 0.7
    
    def fit(self, prompts, labels):
        X = self.embedder.encode(prompts, show_progress_bar=True)
        self.classifier.fit(X, labels)
        return self
    
    def predict(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        probs = self.predict_proba(prompts)
        preds = (probs[:, 1] >= self.threshold).astype(int)
        confidence = np.max(probs, axis=1)
        return preds, confidence, probs
    
    def predict_proba(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        X = self.embedder.encode(prompts)
        return self.classifier.predict_proba(X)
    
    def evaluate(self, prompts, labels):
        preds, conf, _ = self.predict(prompts)
        acc = np.mean(preds == labels)
        high_conf_mask = conf >= self.threshold
        high_conf_acc = np.mean(preds[high_conf_mask] == np.array(labels)[high_conf_mask])
        return {
            'accuracy': acc,
            'high_confidence_accuracy': high_conf_acc,
            'high_confidence_ratio': high_conf_mask.mean()
        }
