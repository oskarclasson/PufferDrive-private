"""A map-editor bin (data/drawn_maps) in place of the online Simian map, for the policy server.

Map-editor coordinates are metres east/north of the spec's lat/lon origin on a sphere (scripts/map_editor);
the Simian map frame is UTM. Positions are converted exactly through WGS84, headings and velocities are
rotated by the angle between the two frames' east axes at the ego start.
"""

import json
import math

import numpy as np

EARTH_RADIUS_M = 6378137.0  # EARTH_RADIUS_M in scripts/map_editor/editor.html
WGS84_A_M = 6378137.0
WGS84_F = 1 / 298.257223563
UTM_SCALE = 0.9996
UTM_FALSE_EASTING_M = 500000.0
UTM_FALSE_NORTHING_SOUTH_M = 10000000.0
UTM_MIN_ZONE = 1
UTM_MAX_ZONE = 60
HEADING_PROBE_M = 10.0


def _krueger():
    n = WGS84_F / (2 - WGS84_F)
    rectifying_radius_m = WGS84_A_M / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    alpha = (n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5, 61 * n**3 / 240)
    beta = (n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96, n**2 / 48 + n**3 / 15, 17 * n**3 / 480)
    delta = (2 * n - 2 * n**2 / 3 - 2 * n**3, 7 * n**2 / 3 - 8 * n**3 / 5, 56 * n**3 / 15)
    return n, rectifying_radius_m, alpha, beta, delta


class UTMProjection:
    """Transverse Mercator on WGS84 (Krueger series, sub-millimetre within a zone)."""

    def __init__(self, zone, north):
        if not UTM_MIN_ZONE <= int(zone) <= UTM_MAX_ZONE:
            raise ValueError(f"UTM zone {zone} outside [{UTM_MIN_ZONE}, {UTM_MAX_ZONE}]")
        self.central_lon_rad = math.radians(6 * int(zone) - 183)
        self.false_northing_m = 0.0 if north else UTM_FALSE_NORTHING_SOUTH_M

    def to_latlon(self, easting_m, northing_m):
        _, radius_m, _, beta, delta = _krueger()
        xi = (np.asarray(northing_m, np.float64) - self.false_northing_m) / (UTM_SCALE * radius_m)
        eta = (np.asarray(easting_m, np.float64) - UTM_FALSE_EASTING_M) / (UTM_SCALE * radius_m)
        xi_p = xi - sum(b * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, b in enumerate(beta, 1))
        eta_p = eta - sum(b * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, b in enumerate(beta, 1))
        chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
        lat = chi + sum(d * np.sin(2 * j * chi) for j, d in enumerate(delta, 1))
        lon = self.central_lon_rad + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
        return np.degrees(lat), np.degrees(lon)

    def from_latlon(self, lat_deg, lon_deg):
        n, radius_m, alpha, _, _ = _krueger()
        lat = np.radians(np.asarray(lat_deg, np.float64))
        dlon = np.radians(np.asarray(lon_deg, np.float64)) - self.central_lon_rad
        e2n = 2 * math.sqrt(n) / (1 + n)
        t = np.sinh(np.arctanh(np.sin(lat)) - e2n * np.arctanh(e2n * np.sin(lat)))
        xi_p = np.arctan2(t, np.cos(dlon))
        eta_p = np.arctanh(np.sin(dlon) / np.sqrt(1 + t**2))
        xi = xi_p + sum(a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p) for j, a in enumerate(alpha, 1))
        eta = eta_p + sum(a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p) for j, a in enumerate(alpha, 1))
        return UTM_FALSE_EASTING_M + UTM_SCALE * radius_m * eta, self.false_northing_m + UTM_SCALE * radius_m * xi


class DrawnMap:
    """Frame conversion between the Simian UTM map frame and a map-editor bin. Traffic controls drawn in the
    editor keep their bin states: Simian lights are not streamed into them."""

    def __init__(self, spec_path, utm_zone, utm_north, ego_x, ego_y):
        with open(spec_path) as spec_file:
            spec = json.load(spec_file)
        origin = spec.get("origin")
        if not origin:
            raise ValueError(f"{spec_path} has no lat/lon origin: set one in the map editor")
        self.origin_lat_deg = float(origin["lat"])
        self.origin_lon_deg = float(origin["lon"])
        self.utm = UTMProjection(utm_zone, utm_north)
        self.metres_per_deg_lat = EARTH_RADIUS_M * math.pi / 180
        self.metres_per_deg_lon = self.metres_per_deg_lat * math.cos(math.radians(self.origin_lat_deg))
        ex, ey = self.to_bin(ego_x, ego_y)
        px, py = self.to_bin(ego_x + HEADING_PROBE_M, ego_y)
        self.heading_offset_rad = math.atan2(float(py - ey), float(px - ex))
        self.lane_idx_by_id = {lane["id"]: lane_idx for lane_idx, lane in enumerate(spec["lanes"])}
        self.element_light_ids = ()

    @property
    def num_traffic(self):
        return len(self.element_light_ids)

    def to_bin(self, x, y):
        lat, lon = self.utm.to_latlon(x, y)
        x_m = (lon - self.origin_lon_deg) * self.metres_per_deg_lon
        return x_m, (lat - self.origin_lat_deg) * self.metres_per_deg_lat

    def from_bin(self, x, y):
        lat = self.origin_lat_deg + np.asarray(y, np.float64) / self.metres_per_deg_lat
        lon = self.origin_lon_deg + np.asarray(x, np.float64) / self.metres_per_deg_lon
        return self.utm.from_latlon(lat, lon)

    def heading_to_bin(self, heading_rad):
        return heading_rad + self.heading_offset_rad

    def velocity_to_bin(self, vx, vy):
        cos_offset, sin_offset = math.cos(self.heading_offset_rad), math.sin(self.heading_offset_rad)
        return vx * cos_offset - vy * sin_offset, vx * sin_offset + vy * cos_offset
