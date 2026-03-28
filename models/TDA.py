import torch
import torch.nn.functional as F


class TDA:
    def __init__(self,text_features,cache_size=1000,k=5,alpha=1.0,beta=5.0,confidence_threshold=0.7,device=None):
        """
        text_features: [num_classes, dim] (normalized)
        """

        self.text_features = F.normalize(text_features, dim=-1)
        self.cache_features = []
        self.cache_labels = []

        self.cache_size = cache_size
        self.k = k
        self.alpha = alpha  # weight for CLIP logits
        self.beta = beta    # weight for cache logits
        self.conf_thresh = confidence_threshold

        self.device = device

    def _get_clip_logits(self, image_features):
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.text_features.T
        return logits

    def _update_cache(self, image_features, pred_label, confidence):
        if confidence < self.conf_thresh:
            return

        if len(self.cache_features) >= self.cache_size:
            self.cache_features.pop(0)
            self.cache_labels.pop(0)

        self.cache_features.append(image_features.detach())
        self.cache_labels.append(pred_label.detach())

    def _compute_cache_logits(self, image_features):
        if len(self.cache_features) == 0:
            return None

        cache_feats = torch.stack(self.cache_features)  # [N, dim]
        cache_labels = torch.stack(self.cache_labels)  # [N]

        # cosine similarity
        sim = image_features @ cache_feats.T  # [1, N]

        # top-k neighbors
        k = min(self.k, sim.shape[1])
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1)

        topk_labels = cache_labels[topk_idx.squeeze(0)]  # [k]

        # softmax weights
        weights = F.softmax(topk_vals, dim=-1)  # [1, k]

        num_classes = self.text_features.shape[0]
        cache_logits = torch.zeros((1, num_classes), device=self.device)

        for i in range(k):
            label = topk_labels[i]
            cache_logits[0, label] += weights[0, i]

        return cache_logits

    def predict(self, image_features):
        """
        image_features: [1, dim]
        """

        image_features = F.normalize(image_features, dim=-1)

        # CLIP logits
        clip_logits = self._get_clip_logits(image_features)

        # Cache logits
        cache_logits = self._compute_cache_logits(image_features)

        if cache_logits is not None:
            final_logits = self.alpha * clip_logits + self.beta * cache_logits
        else:
            final_logits = clip_logits

        probs = F.softmax(final_logits, dim=-1)
        confidence, pred = torch.max(probs, dim=-1)

        # Update cache
        self._update_cache(image_features, pred, confidence)

        return pred, confidence, final_logits