"""Goal inference over the ways ahead of co-sim actors, for the joint prediction of the policy server.

The eval's eval.goal_inference, keyed by the external sim's actor ids instead of env rows: every tracked actor keeps
a belief over the branches of the map's route graph ahead of it, updated each step by exp(beta * advantage) with the
critic's advantage gamma * V(next) - V(now) under that branch. All positions are in the bin frame.
"""

import math

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive import route_graph as drive_route_graph

GOAL_COLUMNS = 4  # x, y, z, lane
GOAL_NEW_BRANCH_MIN_BELIEF = 0.1
GOAL_CONTINUE_M = 3.0
MAX_SPEED_MPS = 20.0  # sizes the goal list so it lasts the whole horizon


def distance_to_path_m(point, path):
    """Distance from a point to the polyline through path's (n, 2) points."""
    starts, ends = path[:-1], path[1:]
    segment = ends - starts
    along = np.clip(
        np.einsum("ij,ij->i", point - starts, segment) / np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-9),
        0.0,
        1.0,
    )
    return float(np.min(np.linalg.norm(starts + along[:, None] * segment - point, axis=1)))


def carry_belief(old_last_goals, old_belief, new_paths):
    """Split each old hypothesis' belief over the new ones continuing it, each new one taking the old hypothesis
    whose farthest goal its path passes closest; forks and new branches start with GOAL_NEW_BRANCH_MIN_BELIEF."""
    if not len(old_last_goals):
        return np.full(len(new_paths), 1.0 / len(new_paths), dtype=np.float32)
    carried = np.zeros(len(new_paths), dtype=np.float32)
    gap_m = np.array([[distance_to_path_m(goal, path) for goal in old_last_goals] for path in new_paths])
    parent = np.where(gap_m.min(axis=1) < GOAL_CONTINUE_M, gap_m.argmin(axis=1), -1)
    children = np.bincount(parent[parent >= 0], minlength=len(old_last_goals))
    continued = parent >= 0
    carried[continued] = old_belief[parent[continued]] / children[parent[continued]]
    carried /= max(carried.sum(), 1e-12)
    fork_branch = ~continued | (children[np.maximum(parent, 0)] > 1)
    carried[fork_branch] = np.maximum(carried[fork_branch], GOAL_NEW_BRANCH_MIN_BELIEF)
    return carried / max(carried.sum(), 1e-12)


class ActorGoals:
    """One actor's goal hypotheses, the goals of each it has reached and its belief over them."""

    def __init__(self, hypothesis_count, goal_count):
        shape = (hypothesis_count,)
        self.goals = np.zeros((hypothesis_count, goal_count, GOAL_COLUMNS), dtype=np.float32)
        self.goal_total = np.zeros(shape, dtype=np.int64)
        self.next_goal = np.zeros(shape, dtype=np.int64)
        self.belief = np.zeros(shape, dtype=np.float32)
        self.value = np.zeros(shape, dtype=np.float32)
        self.has_value = np.zeros(shape, dtype=bool)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.xy = None
        self.steps_since_update = 0

    @property
    def valid(self):
        return self.next_goal < self.goal_total


class GoalInference:
    def __init__(
        self,
        bin_path,
        num_goals,
        hypothesis_count,
        goal_spacing_m,
        update_steps,
        beta,
        forgetting,
        gamma,
        goal_radius_m,
        reward_goal,
        horizon_s,
    ):
        from data_utils import mirror_map_bin

        self.graph = drive_route_graph.RouteGraph(mirror_map_bin.read_bin(bin_path)["roads"])
        self.num_goals = int(num_goals)
        self.hypothesis_count = int(hypothesis_count)
        self.goal_spacing_m = float(goal_spacing_m)
        self.update_steps = int(update_steps)
        self.beta = float(beta)
        self.forgetting = float(forgetting)
        self.gamma = float(gamma)
        self.goal_radius_m = float(goal_radius_m)
        self.reward_goal = float(reward_goal)
        self.extension_goal_count = min(
            binding.MAX_GOALS - self.num_goals, math.ceil(horizon_s * MAX_SPEED_MPS / self.goal_spacing_m)
        )
        self.actors = {}

    def _branches(self, x, y, heading):
        goals = np.zeros((self.hypothesis_count, self.num_goals, GOAL_COLUMNS), dtype=np.float32)
        total = np.zeros(self.hypothesis_count, dtype=np.int64)
        try:
            branches = self.graph.branch_goals(
                x, y, heading, self.goal_spacing_m, self.num_goals, self.hypothesis_count
            )
        except drive_route_graph.RouteError:
            return goals, total
        for k, branch in enumerate(branches):
            goals[k, : len(branch)] = branch
            total[k] = len(branch)
        return goals, total

    def advance(self, actor_id, x, y, heading):
        """Score the goals the actor reached since the last step and re-place its hypotheses when due; returns whether
        it has any way ahead."""
        actor = self.actors.get(actor_id)
        if actor is None:
            actor = ActorGoals(self.hypothesis_count, self.num_goals)
            self.actors[actor_id] = actor
        xy = np.array([x, y], dtype=np.float32)
        valid = actor.valid
        actor.reward[:] = 0.0
        # The env's reach test: the goal lies within the goal radius of the segment driven this step.
        if actor.xy is not None:
            (hypotheses,) = np.nonzero(valid)
            goal_xy = actor.goals[hypotheses, actor.next_goal[hypotheses], :2]
            segment = xy - actor.xy
            along = np.clip((goal_xy - actor.xy) @ segment / max(float(segment @ segment), 1e-9), 0.0, 1.0)
            miss_m = np.linalg.norm(actor.xy + along[:, None] * segment - goal_xy, axis=1)
            reached = hypotheses[miss_m < self.goal_radius_m]
            actor.reward[reached] = self.reward_goal
            actor.next_goal[reached] += 1
        actor.xy = xy
        window_left = np.where(valid, actor.goal_total - actor.next_goal, 0)
        periodic = actor.steps_since_update >= self.update_steps
        actor.steps_since_update += 1
        if periodic or window_left.max() < self.num_goals:
            new_goals, new_total = self._branches(x, y, heading)
            # Near dead ends a reach replan reaches no further; keeping the goals keeps their evidence.
            if periodic or new_total.max() > window_left.max():
                old = np.flatnonzero(valid)
                new = np.flatnonzero(new_total > 0)
                new_paths = [np.vstack([xy, new_goals[k, : new_total[k], :2]]) for k in new]
                old_last_goals = actor.goals[old, actor.goal_total[old] - 1, :2]
                actor.belief[:] = 0.0
                if new_paths:
                    actor.belief[new] = carry_belief(old_last_goals, actor.belief[old], new_paths)
                actor.goals, actor.goal_total = new_goals, new_total
                actor.next_goal[:] = 0
                actor.has_value[:] = False
                actor.steps_since_update = 0
        return bool(actor.valid.any())

    def retain(self, actor_ids):
        """Forget the actors no longer tracked."""
        self.actors = {actor_id: self.actors[actor_id] for actor_id in actor_ids if actor_id in self.actors}

    def windows(self, actor_id):
        """The (hypotheses, num_goals, 4) goal windows ahead of the actor and their (hypotheses,) goal counts."""
        actor = self.actors[actor_id]
        valid = actor.valid
        window_count = np.minimum(actor.goal_total - actor.next_goal, self.num_goals).astype(np.int32)
        window_count[~valid] = 0
        window_idx = np.minimum(actor.next_goal[:, None] + np.arange(self.num_goals), self.num_goals - 1)
        windows = np.take_along_axis(actor.goals, window_idx[..., None], axis=1)
        windows[np.arange(self.num_goals) >= window_count[:, None]] = 0.0
        return windows, window_count

    def update(self, actor_id, value):
        """Bayes update of the actor's belief with the critic values of its (hypotheses,) goal windows."""
        actor = self.actors[actor_id]
        valid = actor.valid
        compared = valid & actor.has_value
        # Boltzmann-rational driver: a move is as likely under a route as it raised that route's return.
        if compared.any():
            advantage = np.where(compared, actor.reward + self.gamma * value - actor.value, -np.inf)
            likelihood = np.exp(self.beta * (advantage - advantage.max()))
            posterior = actor.belief * likelihood
            posterior /= max(posterior.sum(), 1e-12)
            actor.belief = ((1.0 - self.forgetting) * posterior + self.forgetting * valid / valid.sum()).astype(
                np.float32
            )
        actor.value, actor.has_value = value.astype(np.float32), valid
        actor.belief[~valid] = 0.0
        actor.belief /= max(actor.belief.sum(), 1e-12)

    def likeliest_goal_list(self, actor_id):
        """The likeliest window continued along the cheapest way past its last goal so it outlasts the horizon, as an
        (n, 4) goal list."""
        actor = self.actors[actor_id]
        windows, window_count = self.windows(actor_id)
        likeliest = int(np.argmax(actor.belief))
        window = windows[likeliest, : window_count[likeliest]]
        previous_xy = window[-2, :2] if len(window) > 1 else actor.xy
        heading = math.atan2(window[-1, 1] - previous_xy[1], window[-1, 0] - previous_xy[0])
        try:
            branches = self.graph.branch_goals(
                *window[-1, :2], heading, self.goal_spacing_m, self.extension_goal_count, 1
            )
        except drive_route_graph.RouteError:
            return window
        if not branches:
            return window
        return np.vstack([window, np.asarray(branches[0], dtype=np.float32)])
