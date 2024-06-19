
#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Social interaction environment

__author__: Stefano De Giorgis, Nicola Chinchella

"""


import numpy as np
from pymdp.envs.env import Env

class SocialInteraction(Env):
    def __init__(self, num_users=100, num_posts=50):
        self.num_users = num_users
        self.num_posts = num_posts
        self.actions = ["FOLLOW", "LIKE", "COMMENT", "SHARE"]
        
        # State representation: user states and post states
        self.user_states = np.zeros(num_users)
        self.post_states = np.zeros(num_posts)
        
        # Initialize rewards matrix or any other relevant matrices
        self.rewards = np.zeros((num_users, num_posts))
        
        # Other initializations as needed
        self.reset()
        
    def reset(self):
        """
        Reset each agent to zero interactions.
        """
        # Reset user and post states
        self.user_states = np.zeros(self.num_users)
        self.post_states = np.zeros(self.num_posts)
        
        # Reset rewards or other dynamic matrices
        self.rewards = np.zeros((self.num_users, self.num_posts))
        
    def step(self, user_id, post_id, action):
        # Simulate the action taken by the user on a post
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Example dynamics for each action
        if action == "FOLLOW":
            self.user_states[user_id] += 1
            reward = 1
        elif action == "LIKE":
            self.post_states[post_id] += 1
            reward = 0.5
        elif action == "COMMENT":
            self.post_states[post_id] += 2
            reward = 1.5
        elif action == "SHARE":
            self.post_states[post_id] += 3
            reward = 2
        
        # Update rewards
        self.rewards[user_id, post_id] += reward
        
        # Return the new state and reward
        return self.user_states, self.post_states, reward
    
    def get_user_state(self, user_id):
        # Return the state of a specific user
        return self.user_states[user_id]
    
    def get_post_state(self, post_id):
        # Return the state of a specific post
        return self.post_states[post_id]
    
    def get_rewards(self):
        # Return the rewards matrix
        return self.rewards

# Example usage
if __name__ == "__main__":
    env = SocialInteraction()
    print("Initial user states:", env.user_states)
    print("Initial post states:", env.post_states)
    
    new_states, new_posts, reward = env.step(user_id=0, post_id=0, action="LIKE")
    print("New user states:", new_states)
    print("New post states:", new_posts)
    print("Reward for the action:", reward)
    
    new_states, new_posts, reward = env.step(user_id=1, post_id=0, action="COMMENT")
    print("New user states:", new_states)
    print("New post states:", new_posts)
    print("Reward for the action:", reward)
    
    print("Rewards matrix:", env.get_rewards())
