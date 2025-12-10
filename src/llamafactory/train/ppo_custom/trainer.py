# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPO Trainer with Custom Reward Function

This trainer uses a custom reward function defined in `custom_rewards/reward_functions.py`
instead of a trained reward model.

To customize the reward:
    1. Edit `custom_rewards/reward_functions.py`
    2. Modify the `custom_reward` function
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..ppo.trainer import CustomPPOTrainer


if TYPE_CHECKING:
    pass


# Add project root to path for importing custom_rewards
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CustomRewardPPOTrainer(CustomPPOTrainer):
    """
    PPO Trainer that uses a custom reward function instead of a reward model.

    The reward function is defined in `custom_rewards/reward_functions.py`.
    """

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """
        Compute rewards using the custom reward function.

        Args:
            queries: List of query tensors (token ids)
            responses: List of response tensors (token ids)

        Returns:
            List of reward tensors
        """
        # Import custom reward function (import here to allow hot-reloading)
        from custom_rewards import custom_reward

        rewards = []
        for query, response in zip(queries, responses):
            # Decode tokens to text
            query_text = self.tokenizer.decode(query, skip_special_tokens=True)
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)

            # Calculate reward using custom function
            reward_value = custom_reward(query_text, response_text)

            # Convert to tensor
            reward_tensor = torch.tensor([reward_value], dtype=torch.float32)
            rewards.append(reward_tensor)

        return rewards
