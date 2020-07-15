class EpisodeResult:
    def __init__(self, iterations, steps_to_completion, total_solved):
        self.iterations = iterations
        self.steps_to_completion = steps_to_completion
        self.total_solved = total_solved
        self.error = (iterations - total_solved) / iterations

    def __str__(self):
        if bool(self.steps_to_completion):
            return "iterations {}, steps_to_completion {}, total_solved {}, error {}" \
                .format(self.iterations, self.steps_to_completion, self.total_solved, self.error)
        return ""
