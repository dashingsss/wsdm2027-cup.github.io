from algorithms.base import Guidance, PlanningAlgorithm


ADVICE = ('Follow the current task goal and real observations. An available action is not necessarily '
          'useful; avoid undoing requirements already satisfied. Historical examples, if supplied, '
          'are advice only: adapt their objects, IDs and actions to the current task. '
          'Their observations are not evidence obtained in the current task.')


class NoRetrieval(PlanningAlgorithm):
    requires_corpus = False

    def reset(self, initial, corpus):
        pass

    def get_guidance(self, state):
        return Guidance(text=ADVICE)


def create_algorithm(config):
    return NoRetrieval()
