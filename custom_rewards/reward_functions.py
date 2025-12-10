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
Custom Reward Functions for PPO Training

This file contains the custom reward function used by PPO (Custom Reward) training.
Modify the `custom_reward` function below to implement your own reward logic.

Usage:
    1. Select "PPO (Custom Reward)" in the GUI training stage dropdown
    2. Modify the `custom_reward` function below
    3. Start training

The function receives:
    - query (str): The input prompt/question
    - response (str): The model's generated response

The function should return:
    - float: A reward value (can be positive or negative)
"""

import random


def custom_reward(query: str, response: str) -> float:
    """
    Calculate custom reward for a query-response pair.

    Args:
        query: The input prompt/question
        response: The model's generated response

    Returns:
        A reward value (float). Higher values indicate better responses.

    Examples:
        # Example 1: Length-based reward
        reward = len(response) / 100.0

        # Example 2: Keyword-based reward
        reward = 1.0 if "thank you" in response.lower() else 0.0

        # Example 3: External API call
        reward = call_sentiment_api(response)

        # Example 4: Rule-based reward
        reward = 0.0
        if len(response) > 50:
            reward += 0.5
        if "sorry" not in response.lower():
            reward += 0.3
    """
    # ============================================================
    # Custom reward: Exactly 3 words = positive, otherwise negative
    # ============================================================

    # Count words in response
    words = response.split()
    word_count = len(words)

    if word_count == 3:
        reward = 1.0   # Positive reward for exactly 3 words
    else:
        reward = -1.0  # Negative reward for other lengths

    return reward
