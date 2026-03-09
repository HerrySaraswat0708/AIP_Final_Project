import torch


class GDAModel:

    def __init__(self, text_features):

        """
        Initialize GDA parameters

        text_features = CLIP text embeddings
        shape: (num_classes, feature_dim)
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mu = torch.tensor(text_features).float().to(self.device)

        self.num_classes = self.mu.shape[0]
        self.feature_dim = self.mu.shape[1]

        # Shared covariance matrix
        self.Sigma = torch.eye(self.feature_dim).to(self.device)

        # Class prior probabilities
        self.pi = torch.ones(self.num_classes).to(self.device) / self.num_classes

        # Sample counts per class
        self.Ny = torch.ones(self.num_classes).to(self.device)

        # Total sample count
        self.nt = 1

    def mahalanobis_distance(self, x):

        """
        Compute Mahalanobis distance for all classes
        """

        Sigma_inv = torch.inverse(self.Sigma)

        distances = []

        for y in range(self.num_classes):

            diff = x - self.mu[y]

            # dist = diff @ Sigma_inv @ diff.T
            dist = torch.dot(diff, Sigma_inv @ diff)
            distances.append(dist)

        distances = torch.stack(distances)

        return distances

    def gaussian_likelihood(self, x):

        """
        Compute Gaussian likelihood p(x|y)
        """

        Sigma_inv = torch.inverse(self.Sigma)

        likelihoods = []

        for y in range(self.num_classes):

            diff = x - self.mu[y]

            exponent = -0.5 * (diff @ Sigma_inv @ diff.T)

            likelihood = torch.exp(exponent)

            likelihoods.append(likelihood)

        likelihoods = torch.stack(likelihoods)

        return likelihoods

    def posterior_probability(self, x):

        """
        Compute posterior probability P(y|x)
        """

        likelihood = self.gaussian_likelihood(x)

        numerator = self.pi * likelihood

        denominator = torch.sum(numerator)

        posterior = numerator / denominator

        return posterior

    def predict(self, x):

        """
        Predict class using GDA
        """

        posterior = self.posterior_probability(x)

        prediction = torch.argmax(posterior)

        return prediction, posterior

    def update_parameters(self, x, gamma):

        """
        M-step update for Online EM
        """

        x = x.squeeze()

        for y in range(self.num_classes):

            gamma_y = gamma[y]

            self.mu[y] = (
                self.Ny[y] * self.mu[y] + gamma_y * x
            ) / (self.Ny[y] + gamma_y)

            self.Ny[y] = self.Ny[y] + gamma_y

        # Update covariance

        Sigma_new = torch.zeros_like(self.Sigma)

        for y in range(self.num_classes):

            diff = x - self.mu[y]

            Sigma_new += gamma[y] * torch.outer(diff, diff)

        self.Sigma = ((self.nt - 1) * self.Sigma + Sigma_new) / self.nt

        self.nt += 1