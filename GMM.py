import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, n_components, covariance_type='full', tol=1e-3, max_iter=100, n_init=1, reg_covar=1e-6,
                 init_params='kmeans'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.init_params = init_params

    def _initialize_parameters(self, X):
        N, D = X.shape
        if self.init_params == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, n_init=self.n_init).fit(X)
            self.means_ = kmeans.cluster_centers_
            self.weights_ = np.array([np.mean(kmeans.labels_ == i) for i in range(self.n_components)])
        else:
            self.means_ = X[np.random.choice(N, self.n_components, replace=False)]
            self.weights_ = np.ones(self.n_components) / self.n_components

        if self.covariance_type == 'full':
            self.covariances_ = np.array(
                [np.cov(X, rowvar=False) + self.reg_covar * np.eye(D) for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            self.covariances_ = np.array(
                [np.diag(np.var(X, axis=0)) + self.reg_covar for _ in range(self.n_components)])

    def _e_step(self, X):
        N, D = X.shape
        self.resp_ = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov_matrix = self.covariances_[k]
            elif self.covariance_type == 'diag':
                cov_matrix = np.diag(self.covariances_[k])
            norm = multivariate_normal(mean=self.means_[k], cov=cov_matrix, allow_singular=True)
            self.resp_[:, k] = self.weights_[k] * norm.pdf(X)

        self.resp_ /= self.resp_.sum(axis=1, keepdims=True)
        return np.sum(np.log(self.resp_.sum(axis=1)))

    def _m_step(self, X):
        N, D = X.shape
        Nk = self.resp_.sum(axis=0)

        self.weights_ = Nk / N
        self.means_ = np.dot(self.resp_.T, X) / Nk[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            if self.covariance_type == 'full':
                self.covariances_[k] = np.dot(self.resp_[:, k] * diff.T, diff) / Nk[k] + self.reg_covar * np.eye(D)
            elif self.covariance_type == 'diag':
                self.covariances_[k] = (self.resp_[:, k] * diff ** 2).sum(axis=0) / Nk[k] + self.reg_covar

    def fit(self, X):
        best_log_likelihood = -np.inf
        best_params = None

        for _ in range(self.n_init):
            self._initialize_parameters(X)
            log_likelihood = None

            for i in range(self.max_iter):
                prev_log_likelihood = log_likelihood
                log_likelihood = self._e_step(X)
                self._m_step(X)

                if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                    break

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (self.weights_, self.means_, self.covariances_)

        self.weights_, self.means_, self.covariances_ = best_params

    def predict_proba(self, X):
        N, D = X.shape
        resp = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov_matrix = self.covariances_[k]
            elif self.covariance_type == 'diag':
                cov_matrix = np.diag(self.covariances_[k])
            norm = multivariate_normal(mean=self.means_[k], cov=cov_matrix, allow_singular=True)
            resp[:, k] = self.weights_[k] * norm.pdf(X)

        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
