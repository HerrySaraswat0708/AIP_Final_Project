import torch
import torch.nn.functional as F

class FreeTTA:
    def __init__(self, text_features, alpha=0.4, beta=5.5, device=None):
        self.device = device
        
        # Initial Text Features (anchors)
        self.mu = text_features.detach().clone().float().to(self.device)
        self.mu = F.normalize(self.mu, dim=-1)
        
        self.num_classes = self.mu.shape[0]
        self.feature_dim = self.mu.shape[1]
        
        # Online EM Parameters (Eq. 12)
        self.Ny = torch.ones(self.num_classes).to(self.device)
        self.nt = torch.tensor(1.0).to(self.device)
        self.sigma = torch.eye(self.feature_dim).to(self.device) * 0.1
        
        # Hyperparameters
        self.alpha = alpha # Influence of generative model (Eq. 13)
        self.beta = beta   # Entropy weighting factor (Eq. 11)
        

    def get_entropy_weight(self, clip_probs):
        # Eq. 11: H(xt) = -sum(P * log P)
        entropy = -torch.sum(clip_probs * torch.log(clip_probs + 1e-10))
        # Weighting function w(h) = e^(-beta * h)
        weight = torch.exp(-self.beta * entropy)
        return weight

    def update_parameters(self, x, posterior, weight):
        # delta_N = w(h) * gamma
        delta_N = weight * posterior
        new_Ny = self.Ny + delta_N
        
        # mu_y' = (Ny * mu + delta_N * x) / (Ny + delta_N)
        # We use a slight momentum/smoothing to keep the "Anchor" influence
        new_mu = (self.Ny.view(-1, 1) * self.mu + delta_N.view(-1, 1) * x) / (new_Ny.view(-1, 1) + 1e-8)
        
        # Sigma' update from Eq. 12
        # To avoid 2% accuracy, ensure Sigma update is a small moving average
        diff = x - self.mu[torch.argmax(posterior)]
        sample_cov = torch.outer(diff, diff)
        self.sigma = 0.99 * self.sigma + 0.01 * sample_cov
        
        self.mu = F.normalize(new_mu, dim=-1)
        self.Ny = new_Ny

    def predict(self, x, clip_probs, clip_logits):
        x = F.normalize(x, dim=-1)
        
        # 1. Weighting (Eq. 11)
        weight = self.get_entropy_weight(clip_probs)

        # 2. Generative Model Part (Eq. 13)
        # wy = Sigma^-1 * mu_y
        # by = log(Py) - 0.5 * mu_y^T * Sigma^-1 * mu_y
        
        # ADD JITTER for stability - crucial to avoid 2% accuracy
        jitter = torch.eye(self.feature_dim).to(self.device) * 1e-4
        stable_sigma = self.sigma + jitter
        
        # Use linalg.solve for Sigma^-1 * mu (more stable than inverse)
        #
        inv_sigma_mu = torch.linalg.solve(stable_sigma, self.mu.t()) # [feature_dim, num_classes]
        
        # wy^T * F
        w_dot_f = torch.matmul(x, inv_sigma_mu) 
        
        # by term (Eq. 13 definition)
        log_py = torch.log(self.Ny / (self.Ny.sum() + 1e-8))
        mu_sigma_mu = 0.5 * torch.sum(self.mu.t() * inv_sigma_mu, dim=0)
        b_y = log_py - mu_sigma_mu
        
        gen_logits = w_dot_f + b_y
        
        # 3. Final Combined Logits (Eq. 13)
        # Ensure alpha is small (e.g., 0.1) so gen_logits don't overwhelm CLIP
        final_logits = clip_logits + self.alpha * gen_logits
        
        # 4. Update Steps (Eq. 12)
        # Use clip_probs for the update to maintain the "VLM Prior"
        self.update_parameters(x, clip_probs, weight)

        pred = torch.argmax(final_logits)

        return pred




















 # def process_sample(self, x, clip_probs, clip_logits):
    #     # x should be normalized image feature F
    #     x = F.normalize(x, dim=-1)
        
    #     # 1. Compute weight based on CLIP confidence (Eq. 11)
    #     weight = self.get_entropy_weight(clip_probs)

    #     # 2. Generative Model Logits (Probabilistic model part of Eq. 13)
    #     # Using a simplified version of the w_y and b_y logic
    #     # gen_logits corresponds to (w_y.T * F + b_y)
    #     gen_logits = torch.matmul(x, self.mu.t()) 
        
    #     # 3. Final Logits Combination (Eq. 13)
    #     # logits = CLIP_logits + alpha * (Generative_logits)
    #     final_logits = clip_logits + self.alpha * gen_logits
        
    #     # 4. E-Step: Calculate posterior for the update
    #     posterior = torch.softmax(gen_logits, dim=0)
        
    #     # 5. M-Step: Update parameters (Eq. 12)
    #     self.update_parameters(x, posterior, weight)

    #     return torch.argmax(final_logits)

    # fine tune
    # def update_parameters(self, x, posterior, weight):
    #     # delta_N = w(h) * gamma
    #     delta_N = weight * posterior
    #     new_Ny = self.Ny + delta_N
        
    #     # Eq 12: Weighted Mean Update
    #     new_mu = (self.Ny.view(-1, 1) * self.mu + delta_N.view(-1, 1) * x) / (new_Ny.view(-1, 1) + 1e-8)
        
    #     # REFINEMENT: Soft Covariance Update
    #     # Instead of argmax, use the posterior to weight the diffs
    #     # This matches the "EM" nature of the paper more closely
    #     diff = x.unsqueeze(0) - self.mu # [num_classes, feature_dim]
    #     # Weighted outer product sum across all classes
    #     sample_cov = torch.einsum('n,ni,nj->ij', delta_N, diff, diff)
        
    #     # Update sigma with a more standard online EM momentum (e.g., 0.9)
    #     self.sigma = 0.9 * self.sigma + 0.1 * sample_cov
        
    #     self.mu = F.normalize(new_mu, dim=-1)
    #     self.Ny = new_Ny

    # def process_sample(self, x, clip_probs, clip_logits):
    #     x = F.normalize(x, dim=-1)
        
    #     # 1. Weighting (Eq. 11) - Try beta=1.0 for DTD
    #     weight = self.get_entropy_weight(clip_probs)

    #     # 2. Generative Model Part (Eq. 13)
    #     jitter = torch.eye(self.feature_dim).to(self.device) * 1e-3 # Slightly higher jitter
    #     stable_sigma = self.sigma + jitter
        
    #     inv_sigma_mu = torch.linalg.solve(stable_sigma, self.mu.t()) 
        
    #     w_dot_f = torch.matmul(x, inv_sigma_mu) 
        
    #     # Eq 13: Ensure P(y) is calculated from the updated Ny
    #     log_py = torch.log(self.Ny / (self.Ny.sum() + 1e-8))
    #     mu_sigma_mu = 0.5 * torch.sum(self.mu.t() * inv_sigma_mu, dim=0)
    #     b_y = log_py - mu_sigma_mu
        
    #     gen_logits = w_dot_f + b_y
        
    #     # 3. Final Fusion (Eq. 13)
    #     # Check if alpha=0.1 provides better stability than 0.2
    #     final_logits = clip_logits + self.alpha * gen_logits
        
    #     # 4. E-Step: The paper uses the COMBINED predictions to guide the update
    #     # because the generative model becomes more reliable over time.
    #     current_posterior = torch.softmax(final_logits, dim=0)
    #     self.update_parameters(x, current_posterior, weight)

    #     return torch.argmax(final_logits)