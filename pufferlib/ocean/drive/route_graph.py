"""Lane route graph over a map's drivable lanes: shortest routes for the scenario editor and the ways ahead of an
agent for goal inference."""

import heapq
import math

import numpy as np


DRIVABLE_LANE_TYPES = (1, 2)
ROUTE_LANE_MAX_HEADING_DIFF_RAD = math.pi / 2
ROUTE_NODE_SPACING_M = 2.0
ROUTE_SNAP_MAX_M = 6.0
ROUTE_GOAL_CANDIDATES = 8
ROUTE_GOAL_OVERLAP_M = 1.0
ROUTE_LANE_CHANGE_AHEAD_MIN_M = 8.0
ROUTE_LANE_CHANGE_AHEAD_MAX_M = 20.0
ROUTE_LANE_CHANGE_LATERAL_MIN_M = 2.0
ROUTE_LANE_CHANGE_LATERAL_MAX_M = 5.0
ROUTE_LANE_CHANGE_MAX_HEADING_DIFF_RAD = math.radians(20.0)
ROUTE_LANE_CHANGE_PENALTY_M = 25.0  # prefer staying in lane unless changing saves this much distance
ROUTE_DUPLICATE_GOAL_M = 3.0
ROUTE_BRANCH_START_MAX_HEADING_DIFF_RAD = math.pi / 4
ROUTE_GAP_LINK_M = 6.0
ROUTE_GAP_LINK_MAX_HEADING_DIFF_RAD = math.pi / 4


class RouteError(ValueError):
    pass


class RouteGraph:
    """Lane centrelines sampled every ~ROUTE_NODE_SPACING_M, linked along lanes, into successor lanes and across
    to the parallel lane beside them (lane changes, at an extra cost), for shortest-route queries."""

    def __init__(self, roads):
        xs, ys, zs, headings, lane_idxs = [], [], [], [], []
        first_node, last_node = {}, {}
        for road in roads:
            if road["type"] not in DRIVABLE_LANE_TYPES or road["S"] < 2:
                continue
            points = _resample_polyline(np.column_stack([road["x"], road["y"], road["z"]]), ROUTE_NODE_SPACING_M)
            if points is None:
                continue
            deltas = np.diff(points[:, :2], axis=0)
            segment_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
            first_node[road["id"]] = len(xs)
            xs.extend(points[:, 0])
            ys.extend(points[:, 1])
            zs.extend(points[:, 2])
            headings.extend(np.append(segment_headings, segment_headings[-1]))
            lane_idxs.extend([road["id"]] * len(points))
            last_node[road["id"]] = len(xs) - 1
        if not xs:
            raise RouteError("base map has no drivable lanes")
        self.x = np.asarray(xs)
        self.y = np.asarray(ys)
        self.z = np.asarray(zs)
        self.heading = np.asarray(headings)
        self.lane_idx = np.asarray(lane_idxs)

        self.neighbors = [[] for _ in xs]
        for node in range(len(xs) - 1):
            if self.lane_idx[node] == self.lane_idx[node + 1]:
                self._link(node, node + 1, 0.0)
        for road in roads:
            if road["id"] not in last_node:
                continue
            for exit_lane in road["exit_lanes"]:
                if exit_lane in first_node:
                    self._link(last_node[road["id"]], first_node[exit_lane], 0.0)
        self._link_gaps(roads, first_node, last_node)
        self._link_lane_changes()

    def _link(self, src, dst, penalty_m):
        distance_m = math.hypot(self.x[dst] - self.x[src], self.y[dst] - self.y[src])
        self.neighbors[src].append((dst, distance_m + penalty_m))

    def _link_gaps(self, roads, first_node, last_node):
        """Dead-end lane -> lane starting just past its end the same way: hand-drawn maps leave such gaps unlinked."""
        starts = np.asarray(list(first_node.values()))
        for road in roads:
            if road["id"] not in last_node or any(exit_lane in first_node for exit_lane in road["exit_lanes"]):
                continue
            end = last_node[road["id"]]
            gap_m = np.hypot(self.x[starts] - self.x[end], self.y[starts] - self.y[end])
            heading_diff = np.abs(np.angle(np.exp(1j * (self.heading[starts] - self.heading[end]))))
            for start in starts[(gap_m <= ROUTE_GAP_LINK_M) & (heading_diff <= ROUTE_GAP_LINK_MAX_HEADING_DIFF_RAD)]:
                if self.lane_idx[start] != self.lane_idx[end]:
                    self._link(end, int(start), 0.0)

    def _link_lane_changes(self):
        """Node -> node on another lane slightly ahead and one lane over, heading the same way."""
        cell = np.floor(np.column_stack([self.x, self.y]) / ROUTE_LANE_CHANGE_AHEAD_MAX_M).astype(np.int64)
        buckets = {}
        for node, key in enumerate(map(tuple, cell)):
            buckets.setdefault(key, []).append(node)
        buckets = {key: np.asarray(nodes) for key, nodes in buckets.items()}
        for (cx, cy), src_nodes in buckets.items():
            nearby = [buckets.get((cx + dx, cy + dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            dst_nodes = np.concatenate([nodes for nodes in nearby if nodes is not None])
            dx = self.x[dst_nodes][None, :] - self.x[src_nodes][:, None]
            dy = self.y[dst_nodes][None, :] - self.y[src_nodes][:, None]
            cos_h, sin_h = np.cos(self.heading[src_nodes])[:, None], np.sin(self.heading[src_nodes])[:, None]
            ahead = dx * cos_h + dy * sin_h
            lateral = np.abs(-dx * sin_h + dy * cos_h)
            heading_diff = np.abs(
                np.angle(np.exp(1j * (self.heading[dst_nodes][None, :] - self.heading[src_nodes][:, None])))
            )
            valid = (
                (self.lane_idx[dst_nodes][None, :] != self.lane_idx[src_nodes][:, None])
                & (ahead >= ROUTE_LANE_CHANGE_AHEAD_MIN_M)
                & (ahead <= ROUTE_LANE_CHANGE_AHEAD_MAX_M)
                & (lateral >= ROUTE_LANE_CHANGE_LATERAL_MIN_M)
                & (lateral <= ROUTE_LANE_CHANGE_LATERAL_MAX_M)
                & (heading_diff <= ROUTE_LANE_CHANGE_MAX_HEADING_DIFF_RAD)
            )
            for src_pos, dst_pos in zip(*np.nonzero(valid)):
                self._link(int(src_nodes[src_pos]), int(dst_nodes[dst_pos]), ROUTE_LANE_CHANGE_PENALTY_M)

    def _start_nodes(self, x, y, heading, max_heading_diff_rad=ROUTE_LANE_MAX_HEADING_DIFF_RAD):
        """Nearest same-direction nodes not behind the actor, one per lane overlapping there (e.g. at a fork)."""
        dx, dy = self.x - x, self.y - y
        ahead = dx * math.cos(heading) + dy * math.sin(heading)
        heading_diff = np.abs(np.angle(np.exp(1j * (self.heading - heading))))
        dist = np.where((heading_diff < max_heading_diff_rad) & (ahead >= 0.0), np.hypot(dx, dy), np.inf)
        nearest_m = float(dist.min())
        if nearest_m > ROUTE_SNAP_MAX_M:
            raise RouteError("the actor is not on a lane heading its way, so no route can start there")
        return [
            int(node)
            for node in np.argsort(dist)[:ROUTE_GOAL_CANDIDATES]
            if dist[node] <= nearest_m + ROUTE_GOAL_OVERLAP_M
        ]

    def shortest_path(self, start_x, start_y, start_heading, goal_x, goal_y):
        """Node indices of the shortest route from the actor's pose to the lane point nearest (goal_x, goal_y)."""
        starts = self._start_nodes(start_x, start_y, start_heading)
        goal_dist = np.hypot(self.x - goal_x, self.y - goal_y)
        nearest_m = float(goal_dist.min())
        if nearest_m > ROUTE_SNAP_MAX_M:
            raise RouteError(f"the destination is more than {ROUTE_SNAP_MAX_M:g} m from any lane")
        # Lanes overlapping at the clicked spot (e.g. in a junction) all count, but not the lane beside it
        candidates = np.argsort(goal_dist)[:ROUTE_GOAL_CANDIDATES]
        candidates = {int(node) for node in candidates if goal_dist[node] <= nearest_m + ROUTE_GOAL_OVERLAP_M}
        distance = {start: 0.0 for start in starts}
        previous = {}
        frontier = [(0.0, start) for start in starts]
        while frontier:
            cost, node = heapq.heappop(frontier)
            if cost > distance[node]:
                continue
            if node in candidates:
                path = [node]
                while path[-1] in previous:
                    path.append(previous[path[-1]])
                return path[::-1]
            for next_node, step_cost in self.neighbors[node]:
                next_cost = cost + step_cost
                if next_cost < distance.get(next_node, math.inf):
                    distance[next_node] = next_cost
                    previous[next_node] = node
                    heapq.heappush(frontier, (next_cost, next_node))
        raise RouteError("no drivable route from the actor to the destination")

    def route_goals(self, start_x, start_y, start_heading, goal_x, goal_y, spacing_m, max_goal_count):
        """Equally spaced goals, at most spacing_m apart, along the shortest route; the last is the destination."""
        path = self.shortest_path(start_x, start_y, start_heading, goal_x, goal_y)
        points = np.column_stack([self.x[path], self.y[path]])
        cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        total_m = float(cumulative[-1])
        goal_count = max(1, math.ceil(total_m / spacing_m))
        if goal_count > max_goal_count:
            raise RouteError(f"a {total_m:.0f} m route needs over {max_goal_count} goals; increase the spacing")
        # Intermediate goals sit on route nodes (lane centres), never on a lane-change diagonal
        step_m = total_m / goal_count
        goal_nodes = [int(np.argmin(np.abs(cumulative - k * step_m))) for k in range(1, goal_count)] + [len(path) - 1]
        goals = [[round(float(points[node, 0]), 2), round(float(points[node, 1]), 2)] for node in goal_nodes]
        route = [[round(float(px), 2), round(float(py), 2)] for px, py in points]
        return {"goals": goals, "route": route, "length_m": round(total_m, 1)}

    def branch_goals(self, x, y, heading, spacing_m, goal_count, max_branches):
        """Goals along every distinct way ahead: one branch per lane the graph reaches spacing_m * goal_count ahead
        (junction exits and lane changes included, the cheapest way into each), plus dead ends on the way.

        Returns up to max_branches lists of (x, y, z, lane) goals spacing_m apart, cheapest branch first.
        """
        horizon_m = spacing_m * goal_count
        # Inside a junction a crossing lane the agent has already turned away from is no way ahead.
        starts = self._start_nodes(x, y, heading, ROUTE_BRANCH_START_MAX_HEADING_DIFF_RAD)
        cost = {start: 0.0 for start in starts}
        travelled_m = {start: 0.0 for start in starts}
        previous = {}
        ends = {}
        frontier = [(0.0, start) for start in starts]
        while frontier:
            node_cost, node = heapq.heappop(frontier)
            if node_cost > cost[node]:
                continue
            if travelled_m[node] >= horizon_m or not self.neighbors[node]:
                lane = int(self.lane_idx[node])
                if lane not in ends or node_cost < ends[lane][0]:
                    ends[lane] = (node_cost, node)
                continue
            for next_node, step_cost in self.neighbors[node]:
                next_cost = node_cost + step_cost
                if next_cost < cost.get(next_node, math.inf):
                    cost[next_node] = next_cost
                    previous[next_node] = node
                    step_m = math.hypot(self.x[next_node] - self.x[node], self.y[next_node] - self.y[node])
                    travelled_m[next_node] = travelled_m[node] + step_m
                    heapq.heappush(frontier, (next_cost, next_node))
        branches = []
        for _, end in sorted(ends.values()):
            path = [end]
            while path[-1] in previous:
                path.append(previous[path[-1]])
            path = np.asarray(path[::-1])
            along_m = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x[path]), np.diff(self.y[path])))])
            # Exactly spacing_m apart along the path, a lane-change step included, so goals never bunch at its ends.
            goal_s = [
                min(k * spacing_m, along_m[-1])
                for k in range(1, goal_count + 1)
                if k * spacing_m <= along_m[-1] + ROUTE_NODE_SPACING_M
            ]
            goals = [
                (
                    float(np.interp(s, along_m, self.x[path])),
                    float(np.interp(s, along_m, self.y[path])),
                    float(np.interp(s, along_m, self.z[path])),
                    int(self.lane_idx[path[min(np.searchsorted(along_m, s), len(path) - 1)]]),
                )
                for s in goal_s
            ]
            if goals and not any(_same_goals(goals, kept) for kept in branches):
                branches.append(goals)
            if len(branches) == max_branches:
                break
        return branches


def _resample_polyline(points, spacing_m):
    """Rows every ~spacing_m of xy arc length along the polyline, every column interpolated and both ends kept; None
    for a degenerate polyline."""
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points[:, :2], axis=0).T))])
    if cumulative[-1] < 1e-3:
        return None
    sample_s = np.linspace(0.0, cumulative[-1], max(2, math.ceil(cumulative[-1] / spacing_m) + 1))
    return np.column_stack([np.interp(sample_s, cumulative, points[:, column]) for column in range(points.shape[1])])


def _same_goals(goals, other_goals):
    """Two branches whose goals all lie within ROUTE_DUPLICATE_GOAL_M, such as overlapping lanes."""
    if len(goals) != len(other_goals):
        return False
    return all(math.hypot(a[0] - b[0], a[1] - b[1]) < ROUTE_DUPLICATE_GOAL_M for a, b in zip(goals, other_goals))
