"""Goal inference over the ways ahead of co-sim actors, for the joint prediction of the policy server.

The eval's eval.goal_inference, keyed by the external sim's actor ids instead of env rows: every tracked actor keeps
a belief over the branches of the map's route graph ahead of it, Bayes-updated each step by the likelihood of its last
move under that branch: with LIKELIHOOD_ACTION a Gaussian on the jerk it drove around the policy's mean jerk under the
branch, with LIKELIHOOD_VALUE exp(beta * advantage) with the critic's advantage gamma * V(next) - V(now). All positions
are in the bin frame.
"""

import math

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive import route_graph as drive_route_graph

GOAL_COLUMNS = 4  # x, y, z, lane
GOAL_NEW_BRANCH_MIN_BELIEF = 0.1
GOAL_CONTINUE_M = 3.0
MAX_SPEED_MPS = 20.0  # sizes the goal list so it lasts the whole horizon
LIKELIHOOD_VALUE = "value"
LIKELIHOOD_ACTION = "action"
LATERAL_ACCEL_MIN_SPEED_MPS = 1.0  # drive.h's v_eff floor turning lateral acceleration into curvature


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
        self.mean_jerk = np.zeros((hypothesis_count, 2), dtype=np.float32)
        self.has_mean_jerk = np.zeros(shape, dtype=bool)
        self.xy = None
        self.heading = None
        self.speed_mps = None
        self.time_s = None
        self.accel = None  # (longitudinal, lateral) [m/s^2]
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
        likelihood,
        jerk_std,
    ):
        from data_utils import mirror_map_bin

        if likelihood not in (LIKELIHOOD_VALUE, LIKELIHOOD_ACTION):
            raise ValueError(f"unknown goal inference likelihood {likelihood!r}")

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
        self.likelihood = likelihood
        self.jerk_std = np.asarray(jerk_std, dtype=np.float32)  # (longitudinal, lateral) [m/s^3]
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

    def advance(self, actor_id, x, y, heading, speed_mps, time_s):
        """Score the goals the actor reached since the last step, score its hypotheses by the jerk it drove with
        LIKELIHOOD_ACTION, and re-place them when due; returns whether it has any way ahead. speed_mps is signed along
        heading."""
        actor = self.actors.get(actor_id)
        if actor is None:
            actor = ActorGoals(self.hypothesis_count, self.num_goals)
            self.actors[actor_id] = actor
        xy = np.array([x, y], dtype=np.float32)
        valid = actor.valid
        # Before any replan: the cached mean jerks belong to the hypotheses the actor drove under.
        self._observe_motion(actor, heading, speed_mps, time_s, valid)
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
                old_belief = actor.belief[old]
                actor.belief[:] = 0.0
                if new_paths:
                    actor.belief[new] = carry_belief(old_last_goals, old_belief, new_paths)
                actor.goals, actor.goal_total = new_goals, new_total
                actor.next_goal[:] = 0
                actor.has_value[:] = False
                actor.has_mean_jerk[:] = False
                actor.steps_since_update = 0
        return bool(actor.valid.any())

    def _observe_motion(self, actor, heading, speed_mps, time_s, valid):
        """Estimate the actor's acceleration from its speed and heading since the last step and, with
        LIKELIHOOD_ACTION, Bayes-update its belief by a Gaussian on the jerk since then around each hypothesis' mean
        jerk."""
        if actor.time_s is not None:
            dt_s = time_s - actor.time_s
            if not dt_s > 0:
                raise ValueError(f"actor seen at {time_s} s after {actor.time_s} s")
            yaw_rate = math.remainder(heading - actor.heading, 2 * math.pi) / dt_s
            curvature = yaw_rate / math.copysign(max(abs(speed_mps), LATERAL_ACCEL_MIN_SPEED_MPS), speed_mps)
            accel = np.array(
                [
                    (speed_mps - actor.speed_mps) / dt_s,
                    curvature * max(abs(speed_mps), LATERAL_ACCEL_MIN_SPEED_MPS) ** 2,
                ],
                dtype=np.float32,
            )
            scored = valid & actor.has_mean_jerk
            if self.likelihood == LIKELIHOOD_ACTION and actor.accel is not None and scored.any():
                jerk = (accel - actor.accel) / dt_s
                jerk_error = (actor.mean_jerk - jerk) / self.jerk_std
                self._bayes_update(actor, -0.5 * np.sum(jerk_error**2, axis=-1), scored, valid)
            actor.accel = accel
        actor.heading, actor.speed_mps, actor.time_s = heading, speed_mps, time_s

    def _bayes_update(self, actor, log_likelihood, scored, valid):
        """Bayes-update the belief over the scored hypotheses, then mix it back toward uniform over the valid ones by
        forgetting."""
        log_likelihood = np.where(scored, log_likelihood, -np.inf)
        posterior = actor.belief * np.exp(log_likelihood - log_likelihood.max())
        posterior /= max(posterior.sum(), 1e-12)
        actor.belief = ((1.0 - self.forgetting) * posterior + self.forgetting * valid / valid.sum()).astype(np.float32)

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

    def update(self, actor_id, value, mean_jerk):
        """Take the critic values and the policy's (hypotheses, 2) mean jerks [m/s^3] of the actor's (hypotheses,)
        goal windows: with LIKELIHOOD_VALUE Bayes-update its belief by the values, with LIKELIHOOD_ACTION keep the
        mean jerks to score its next move against."""
        actor = self.actors[actor_id]
        valid = actor.valid
        if self.likelihood == LIKELIHOOD_ACTION:
            actor.mean_jerk, actor.has_mean_jerk = mean_jerk.astype(np.float32), valid
        else:
            compared = valid & actor.has_value
            # Boltzmann-rational driver: a move is as likely under a route as it raised that route's return.
            if compared.any():
                advantage = actor.reward + self.gamma * value - actor.value
                self._bayes_update(actor, self.beta * advantage, compared, valid)
            actor.value, actor.has_value = value.astype(np.float32), valid
        actor.belief[~valid] = 0.0
        actor.belief /= max(actor.belief.sum(), 1e-12)

    def likeliest_goal_list(self, actor_id):
        """The likeliest window continued along the cheapest way past its last goal so it outlasts the horizon, as an
        (n, 4) goal list."""
        return self.goal_list(actor_id, int(np.argmax(self.actors[actor_id].belief)))

    def routes(self, actor_id, min_belief, max_routes):
        """The hypotheses of the actor the robust rollout drives it along, likeliest first: at most max_routes with
        belief of at least min_belief, and always its likeliest, as the eval's robust prediction keeps them."""
        actor = self.actors[actor_id]
        _, window_count = self.windows(actor_id)
        order = [int(k) for k in np.argsort(-actor.belief, kind="stable") if window_count[k] > 0]
        kept = [k for k in order[:max_routes] if actor.belief[k] >= min_belief]
        return kept or order[:1]

    def belief(self, actor_id, hypothesis):
        return float(self.actors[actor_id].belief[hypothesis])

    def goal_list(self, actor_id, hypothesis):
        """The window of a hypothesis continued along the cheapest way past its last goal so it outlasts the horizon,
        as an (n, 4) goal list."""
        actor = self.actors[actor_id]
        windows, window_count = self.windows(actor_id)
        window = windows[hypothesis, : window_count[hypothesis]]
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
