from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningInput:
    """Framework-owned task state, supplied again before each model call.

    goal is the public instruction/question; observation is the latest real
    tool observation (initial observation before any action). history contains
    attempted action strings, valid flags and real returns, including failures.
    available_actions may be incomplete for free-text search/finish tools.
    Current task reference answers and model credentials are not inputs.
    Treat this object and its nested history dictionaries as read-only.
    Docker delivery uses JSON, not shared Python objects. Runtime status and
    counters remain compatible; public corpus files stay cleaned.
    """
    task_id: str
    goal: str
    observation: str
    available_actions: tuple[str, ...]
    history: tuple[dict, ...]
    domain: str = "alfworld"


@dataclass(frozen=True)
class Guidance:
    """Prompt advice and the IDs of public trajectories used to produce it."""
    text: str = ""
    retrieved_ids: tuple[str, ...] = ()


class PlanningAlgorithm(ABC):
    # Constructors/factories must be lightweight; build indexes in prepare.
    requires_corpus = True

    def prepare(self, corpus: list[dict]) -> None:
        """Optional preparation before episodes; the default does nothing.

        The framework loads corpus rows (task_id, goal, actions, observations,
        initial_observation and steps). Do not mutate these shared rows.
        This runs in the algorithm container once per domain, except recovery.
        The same corpus object is reused by reset; do not rebuild it per task.
        No preparation deadline is imposed; organizers may stop stalled jobs.
        A method may build an in-memory index here and update its own state
        later in get_guidance; no prebuilt index upload is required.
        Factories receive seed, domain and options, never credentials.
        The framework does not impose a reference-count limit.
        """

    @abstractmethod
    def reset(self, initial: PlanningInput, corpus: list[dict]) -> None:
        """Start a task; clear prior task state, retaining public-corpus indexes.

        This is a reset contract, not OS-level isolation between test tasks.
        The service persists across tasks to reuse its public index.
        Static baselines select their reference here. Dynamic methods may wait
        until get_guidance. No environment reset is performed by this method.
        """

    @abstractmethod
    def get_guidance(self, state: PlanningInput) -> Guidance:
        """Called before each LLM request; return Guidance, not a tool action.

        May reuse cached advice, retrieve again, or update task-local indexes.
        Repeated calls are possible without a new tool action: methods that
        update an index must avoid ingesting the same history entry twice.
        """
