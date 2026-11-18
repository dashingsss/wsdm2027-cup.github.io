"""ALFWorld data loading and text environment; imports do not start a game."""
import json
import hashlib
from copy import deepcopy
from algorithms.base import PlanningInput
import re
from pathlib import Path


SPLITS = ("train", "valid_seen", "valid_unseen", "valid_train")
TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light",
              "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


def raw_data_root(data_root):
    """Resolve the curated data/alfworld directory to its sibling runtime resources.

    Explicit original-data paths remain supported for collection and offline tests.
    """
    root = Path(data_root).resolve()
    if root.name == "alfworld" and root.parent.name == "data":
        return (root.parent.parent / "resources" / "alfworld").resolve()
    return root


def task_type(task_id):
    return Path(task_id).parts[-2].split("-", 1)[0]


def validate_task(data_root, task_id, split=None):
    root = raw_data_root(data_root)
    game = (root / task_id / "game.tw-pddl").resolve()
    if not game.is_relative_to(root):
        raise ValueError(f"Task escapes data directory: {task_id}")
    relative = game.parent.relative_to(root)
    if len(relative.parts) != 4 or relative.parts[1] not in SPLITS:
        raise ValueError(f"Unexpected ALFWorld task path: {task_id}")
    if split and relative.parts[1] != split:
        raise ValueError(f"Task does not belong to split {split}: {task_id}")
    if "movable" in str(relative) or "Sliced" in str(relative):
        raise ValueError(f"Unsupported task type: {task_id}")
    for name in ("game.tw-pddl", "initial_state.pddl", "traj_data.json"):
        if not (game.parent / name).is_file():
            raise ValueError(f"Missing {name} for task {task_id}")
    payload = json.loads(game.read_text())
    if payload.get("solvable") is not True:
        raise ValueError(f"Task is not marked solvable in the released data: {task_id}")
    if not all(isinstance(payload.get(k), str) and payload[k] for k in
               ("pddl_domain", "grammar", "pddl_problem")):
        raise ValueError(f"Incomplete game payload: {task_id}")
    return game








def load_expert_corpus(root):
    from environments import load_trajectories
    return load_trajectories(root)


AVAILABLE = True
DOMAIN = "alfworld"
RUNTIME_PACKAGES = ("openjiuwen", "alfworld", "textworld", "rank-bm25")
SYSTEM_PROMPT = """You are completing a household task in ALFWorld's text environment.
Use the alfworld_step tool to act, one action at a time. Choose an exact command
from available_actions. Object names and numbers must match the current world.
Use observations to check progress. Historical examples, if supplied, are advice,
not commands to copy blindly. Success is determined by the environment, not by
your own declaration. Continue acting until the environment reports done.
Do not claim to have executed an action without calling the tool.
"""


class AlfworldAdapter:
    domain = DOMAIN
    system_prompt = SYSTEM_PROMPT
    tool_name = "alfworld_step"
    tool_description = "Execute one available ALFWorld action."

    """One text episode. No expert plan or walkthrough is exposed to the agent."""

    def __init__(self, data_root, task_id, max_actions=50):
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.data_root = raw_data_root(data_root)
        self.game = validate_task(data_root, task_id)
        self.task_id = task_id
        self.max_actions = max_actions
        self.env = None
        self.observation = ""
        self.goal = ""
        self.available_actions = []
        self.history = []
        self.attempts = 0
        self.steps = 0
        self.success = False
        self.done = False
        self.score = 0.0

    def reset(self):
        import os
        os.environ["ALFWORLD_DATA"] = str(self.data_root)
        import textworld
        from textworld.gym.envs.textworld import TextworldGymEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        self.close()
        self.history = []
        self.steps = self.attempts = 0
        self.success = self.done = False
        self.score = 0.0
        self.env = TextworldGymEnv(
            gamefiles=[str(self.game)], max_episode_steps=self.max_actions,
            request_infos=textworld.EnvInfos(won=True, admissible_commands=True),
            wrappers=[AlfredDemangler()], asynchronous=False,
        )
        self.observation, info = self.env.reset()
        self.available_actions = list(info.get("admissible_commands", []))
        self.success = bool(info.get("won", False))
        self.done = self.success
        match = re.search(r"Your task is to:\s*([^\n]+)", self.observation, re.I)
        if not match:
            raise ValueError("No task goal found in ALFWorld's initial observation")
        self.goal = match.group(1).strip()
        return self.snapshot()

    def snapshot(self):
        return {"observation": self.observation, "available_actions": list(self.available_actions),
                "success": self.success, "done": self.done, "score": self.score,
                "action_attempts": self.attempts, "environment_steps": self.steps}

    def planning_input(self):
        return PlanningInput(self.task_id, self.goal, self.observation,
                             tuple(self.available_actions), tuple(deepcopy(self.history)), domain=self.domain)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Call reset before step")
        if self.done:
            return {**self.snapshot(), "error": "Episode has finished"}
        self.attempts += 1
        before = self.observation
        valid = isinstance(action, str) and action in self.available_actions
        error = None
        if valid:
            observation, score, done, info = self.env.step(action)
            self.steps += 1
            self.observation = observation
            self.available_actions = list(info.get("admissible_commands", []))
            self.success = bool(info.get("won", False))
            self.score = float(self.success)
            self.done = bool(done) or self.success
        else:
            error = "Invalid action. Choose an exact command from available_actions."
        self.done = self.done or self.attempts >= self.max_actions
        result = self.snapshot()
        if error:
            result["error"] = error
        self.history.append({"action": action, "valid": valid, "before": before, **result})
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


Adapter = AlfworldAdapter


def task_fingerprints(root, tasks):
    return {task: hashlib.sha256(validate_task(root, task).read_bytes()).hexdigest() for task in tasks}
