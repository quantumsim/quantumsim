import warnings
import numpy as np


class BiasedSampler:
    '''A sampler that returns a uniform choice but with probabilities weighted
    as p_twiddle=p^alpha/Z, with Z a normalisation constant. Also allows for
    readout error to be input when the sampling is called.

    All the class does is to store the product of all p_twiddles for
    renormalisation purposes
    '''

    def __init__(self, readout_error, alpha, rng):
        '''
        @alpha: number between 0 and 1 for renormalisation purposes.
        '''
        self.readout_error = readout_error
        self.alpha = alpha
        self.rng = _ensure_rng(rng)

        self._p_twiddle = 1

        ro_temp = readout_error ** self.alpha
        self._ro_renormalized = ro_temp / \
            (ro_temp + (1 - readout_error)**self.alpha)

    def __next__(self):
        pass

    def send(self, ps):
        '''
        @readout_error: probability of the state update and classical output disagreeing
        '''

        if ps is None:
            return None

        prob_0, prob_1 = ps

        # renormalise probability values
        prob_sum = prob_0 + prob_1
        p0_temp = (prob_0 / prob_sum)**self.alpha
        p1_temp = (prob_1 / prob_sum)**self.alpha
        renorm_prob_0 = p0_temp / (p0_temp + p1_temp)

        rand_0, rand_1 = self.rng.random_sample(2)

        if rand_0 < renorm_prob_0:
            proj_state = 0
            self._p_twiddle *= renorm_prob_0
        else:
            proj_state = 1
            self._p_twiddle *= (1 - renorm_prob_0)

        if rand_1 < self._ro_renormalized:
            declared_state = 1 - proj_state
            cond_prob = self.readout_error
            self._p_twiddle *= self._ro_renormalized
        else:
            declared_state = proj_state
            cond_prob = 1 - self.readout_error
            self._p_twiddle *= (1 - self._ro_renormalized)

        return declared_state, proj_state, cond_prob


def _ensure_rng(rng):
    """Takes random number generator (RNG) or seed as input and instantiates
    and returns RNG, initialized by seed, if it is provided.
    """
    if not hasattr(rng, 'random_sample'):
        if not rng:
            warnings.warn('No random number generator (or seed) provided, '
                          'computation will not be reproducible.')
        # Assuming that we have seed provided instead of RNG
        rng = np.random.RandomState(seed=rng)
    return rng
