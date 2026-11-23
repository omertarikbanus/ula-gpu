"""Stable-Baselines3 policy with separate actor and critic inputs."""

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class ActorCriticInputs(MlpExtractor):
    def __init__(self, critic_dim, actor_dim, net_arch, activation_fn, device):
        super().__init__(critic_dim, net_arch, activation_fn, device)
        self.policy_net = MlpExtractor(actor_dim, {"pi": net_arch.get("pi", [])},
                                      activation_fn, device).policy_net
        self.actor_dim = actor_dim

    def forward_actor(self, features):
        return super().forward_actor(features[..., :self.actor_dim])


class Go2Policy(ActorCriticPolicy):
    def __init__(self, *args, actor_obs_dim=54, **kwargs):
        self.actor_obs_dim = actor_obs_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorCriticInputs(self.features_dim, self.actor_obs_dim,
                                              self.net_arch, self.activation_fn, self.device)
