
class DQN(object):
    def __init__(self, get_model_f):
        self.policy_model = get_model_f()
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.reset_explotation()
        self.random_action_f = random_action_f