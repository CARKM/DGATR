import numpy as np

from policy.qroute import *


class PolicyGradient(Policy):
    attrs = Policy.attrs | set(['Theta', 'discount'])

    def __init__(self, network, initP=0, add_entropy=False, discount=0.9):
        super().__init__(network)
        self.add_entropy = add_entropy
        self.discount = discount
        self.Theta = {x: np.full((len(self.links), len(ys)), initP, dtype=np.float64)
                      for x, ys in self.links.items()}

    def _softmax(self, source, dest):
        e_theta = np.exp(self.Theta[source][dest])
        return e_theta/e_theta.sum()

    def choose(self, source, dest, prob=None):
        """ choose returns the choice following weighted random sample """
        return np.random.choice(self.links[source], p=(
            self._softmax(source, dest) if prob is None else prob))

    def _gradient(self, source, dest, action_idx, softmax=None):
        """ gradient returns a vector with length of neighbors of source """
        gradient = -self._softmax(source, dest) \
            if softmax is None else (-softmax)
        gradient[action_idx] += 1
        return gradient

    def _update_theta(self, r, x, y, dest, max_Q_y, max_Q_x_d, lrp, softmax=None):
        gradient = self._gradient(
            x, dest, self.action_idx[x][y], softmax=softmax)
        self.Theta[x][dest] += lrp * gradient * \
            (r + self.discount * max_Q_y - max_Q_x_d)

    def _update_entropy(self, r, lr, softmax):
        return r - lr * (softmax * np.log2(softmax)).sum()

class HybridQ(PolicyGradient, Qroute):
    attrs = Qroute.attrs | PolicyGradient.attrs

    def __init__(self, network, initQ=0, initP=0, add_entropy=True, discount=0.99):
        PolicyGradient.__init__(self, network, initP,
                                add_entropy=add_entropy, discount=discount)
        Qroute.__init__(self, network, initQ, discount=discount)

    def get_info(self, source, action, packet):
        return {
            'max_Q_y': self.Qtable[action][packet.dest].max(),
            'max_Q_x_d': self.Qtable[source][packet.dest].max(),
        }

    def _update(self, reward, lr={'q': 0.1, 'p': 0.1, 'e': 0.1}):
        r, info, x, y, dest = self._extract(reward)
        softmax = self._softmax(x, dest)
        if self.add_entropy:
            r = self._update_entropy(r, lr['e'], softmax)
        self._update_qtable(r, x, y, dest, info['max_Q_y'], lr['q'])
        self._update_theta(
            r, x, y, dest, info['max_Q_y'], info['max_Q_x_d'], lr['p'], softmax=softmax)


class HybridCQ(PolicyGradient, CQ):
    attrs = CQ.attrs | PolicyGradient.attrs

    def __init__(self, network, initQ=0, initP=0, decay=0.9, add_entropy=False, discount=0.99):
        PolicyGradient.__init__(self, network, initP,
                                add_entropy=add_entropy, discount=discount)
        CQ.__init__(self, network, decay=decay,
                      initQ=initQ, discount=discount)

    def choose(self, source, dest, prob=None, idx = False):
        """ choose returns the choice following weighted random sample """
        if idx == False:
            return np.random.choice(self.links[source], p=(
                self._softmax(source, dest) if prob is None else prob))
        if idx == True:
            scores = self.Qtable[source][dest]
            return (np.argmax(scores), scores.max())


    def get_info(self, source, action, packet):
        z_f, max_Q_f = self.choose(action, packet.dest, idx=True)
        return {
            'max_Q_f': max_Q_f,
            'C_f': self.confidence[action][packet.dest][z_f],
            'max_Q_x_d': self.Qtable[source][packet.dest].max(),
        }

    def _update(self, reward, lr={'p': 0.1, 'e': 0.1}):
        r_f, info, x, y, dest = self._extract(reward)
        softmax_f = self._softmax(x, dest)
        if self.add_entropy:
            r_f = self._update_entropy(r_f, lr['e'], softmax_f)
        self._update_qtable(r_f, x, y, dest, info['C_f'], info['max_Q_f'])
        self._update_theta(
            r_f, x, y, dest, info['max_Q_f'], info['max_Q_x_d'], lr['p'], softmax=softmax_f)


class HybridCDRQ(PolicyGradient, CDRQ):
    attrs = CDRQ.attrs | PolicyGradient.attrs

    def __init__(self, network, add_entropy=False, initQ=0, initP=0, decay=0.9, discount=0.99):
        PolicyGradient.__init__(self, network, initP,
                                add_entropy=add_entropy, discount=discount)
        CDRQ.__init__(self, network, decay=decay,
                      initQ=initQ, discount=discount)

    def get_info(self, source, action, packet):
        w_idx, max_Q_b = Qroute.choose(self, source, packet.source, score=True)
        z_idx, max_Q_f = Qroute.choose(self, action, packet.dest, score=True)
        return {
            'max_Q_b': max_Q_b,
            'max_Q_f': max_Q_f,
            'C_b': self.confidence[source][packet.source][w_idx],
            'C_f': self.confidence[action][packet.dest][z_idx],
            'max_Q_x_d': self.Qtable[source][packet.dest].max(),
            'max_Q_y_s': self.Qtable[action][packet.source].max(),
        }

    def _update(self, reward, lr={'f': 0.85, 'b': 0.95, 'p': 0.1, 'e': 0.1}):
        r_f, info, x, y, dst = self._extract(reward)
        src = reward.packet.source
        r_b = -info['q_x'] - info['t_x']
        softmax_f = self._softmax(x, dst)
        softmax_b = self._softmax(y, src)
        if self.add_entropy:
            r_f = self._update_entropy(r_f, lr['e'], softmax_f)
            r_b = self._update_entropy(r_b, lr['e'], softmax_b)
        # forward update
        self._update_qtable(r_f, x, y, dst, info['C_f'], info['max_Q_f'])
        self._update_theta(
            r_f, x, y, dst, info['max_Q_f'], info['max_Q_x_f'], lr['p'], softmax=softmax_f)
        # backward update
        self._update_qtable(r_b, y, x, src, info['C_b'], info['max_Q_b'])
        self._update_theta(
            r_b, y, x, src, info['max_Q_b'], info['max_Q_y_s'], lr['p'], softmax=softmax_b)
