class EpisodeResult:
    def __init__(self, episodes, steps_to_completion, total_solved, episode_reward, distance_to_goal):
        self.episodes = episodes
        self.steps_to_completion = steps_to_completion
        self.total_solved = total_solved
        self.episode_reward = episode_reward
        self.error = (episodes - total_solved) / episodes
        self.distance_to_goal = distance_to_goal

    def __str__(self):
        if bool(self.steps_to_completion):
            return "episodes {}, steps_to_completion {}, total_solved {}, error {}" \
                .format(self.episodes, self.steps_to_completion, self.total_solved, self.error,)
        return ""
