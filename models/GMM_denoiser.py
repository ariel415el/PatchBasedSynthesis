import joblib
import torch as pt
pt.set_grad_enabled(False)

class GMMDenoiser:
    """
    A denoiser class that uses a GMM in order to denoise the patches
    """

    def __init__(self, pi, mu, sigma, device, MAP: bool = True):
        """
        Initializes the GMM denoiser
        :param gmm: the GMM model which should be used in order to denoise patches
        :param MAP: whether MAP denoisings are to be returned or posterior samples
        """
        super(GMMDenoiser).__init__()
        self.pi = pi.to(device)
        self.mu = mu.to(device)
        self.S = sigma.to(device)
        self._d = mu.shape[1]
        self._calculate_evd()
        self._MAP = MAP
        self.name = f"GMM_k=10_Map={self._MAP}"

    def _calculate_evd(self):
        _precision = 1e-10
        evd = pt.linalg.eigh(self.S.double())
        vals = pt.clamp(evd[0], min=(.5 / 255) ** 2)
        vecs = evd[1]
        self.evd = [vals.double(), vecs.double()]
        self.S = (vecs @ pt.diag_embed(vals) @ vecs.transpose(-2, -1)).double()
        self.L = pt.linalg.cholesky(self.S + _precision * pt.eye(self._d, device=self.mu.device)[None, :, :])

    def _resp(self, y, sig: float):
        """
        Calculates the responsibility of each cluster in the GMM for each patch to be denoised
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :return: the responsibilities of the GMM clusters, as a torch tensor of shape [k, N], where k is the number of
                 clusters in the GMM
        """
        L, U = self.evd
        L = L.to(y.device)
        U = U.to(y.device)
        L = L.clone() + sig
        det = pt.sum(pt.log(L), dim=-1)
        meaned = (y[None] - self.mu[:, None]) @ U
        mahala = pt.sum(meaned * (meaned @ pt.diag_embed((1 / L))), dim=-1).transpose(0, 1)
        ll = pt.log(self.pi[None]) - .5 * (mahala + det[None])
        return pt.exp(ll - pt.logsumexp(ll, dim=1)[:, None])

    def _map(self, y, sig: float, ks: list):
        """
        Returns the (approximate) MAP denoisings of each patch using the GMM model
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :param ks: a list of clusters to be used in order to denoise the patches (a list of length N)
        :return: the denoised patches
        """
        means = sig * self.mu[ks][..., None] + self.S[ks] @ y[..., None]
        L, U = self.evd
        L = L.clone() + sig
        return ((U @ pt.diag_embed(1 / L) @ U.transpose(-2, -1))[ks] @ means)[..., 0]

    def _sample(self, y, sig: float, ks: list):
        """
        Samples denoisings from the GMM
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param sig: the noise variance to remove
        :param ks: a list of clusters to be used in order to denoise the patches (a list of length N)
        :return: the denoised patches
        """
        m = self._map(y, sig, ks)
        L, U = self.evd
        L = 1 / L.clone() + 1 / sig
        samp = (pt.diag_embed(1 / pt.sqrt(L)) @ U.transpose(-2, -1))[ks] @ pt.randn(*m.shape, 1, device=y.device)
        return m + samp[..., 0]

    def denoise(self, y, noise_var: float):
        """
        Main denoising function
        :param y: a torch tensor with shape [N, p_sz, p_sz, 3] containing N different patches which should be denoised
        :param noise_var: the variance of the noise that should be removed from the patches
        :return: the denoised patches, as a torch tensor of shape [N, p_sz, p_sz, 3]
        """
        shp = y.shape
        y = y.clone().reshape(y.shape[0], -1)
        r = self._resp(y, noise_var)

        # return MAP denoisings
        if self._MAP:
            ks = pt.argmax(r, dim=1).tolist()
            return self._map(y, noise_var, ks).reshape(shp)
        # sample patch denoisings from the posterior
        else:
            ks = pt.multinomial(r, 1)[:, 0].tolist()
            return self._sample(y, noise_var, ks).reshape(shp)

    @staticmethod
    def load_from_file(path, **kwargs):
        d = joblib.load(path)
        return GMMDenoiser(mu=d['mu'], pi=d['pi'], sigma=d['Sigma'], **kwargs)

    def save(self, path):
        d = {
            'pi': self.pi,
            'mu': self.mu,
            'Sigma': self.S,
        }
        joblib.dump(d, path)
