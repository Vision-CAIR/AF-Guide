from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GuidedSAC(SAC):
    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.critic_guided = self.policy.critic_guided


class GuidedSACPolicy(SACPolicy):

    def make_critic_guided(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        critic_kwargs['n_critics'] = 1
        critic_kwargs = self._update_features_extractor(critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        if self.share_features_extractor:
            self.critic_guided = self.make_critic_guided(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic_guided.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic_guided = self.make_critic_guided(features_extractor=None)
            critic_parameters = self.critic_guided.parameters()

        self.critic_guided.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)




