"use strict";
// Web-map tiles under a local metric frame (x east, y north of a lat/lon origin), shared by both editors.
const BASEMAP_SOURCES = {
  osm: {url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", max_zoom: 19, attribution: "\u00a9 OpenStreetMap contributors"},
  satellite: {url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", max_zoom: 19, attribution: "Imagery \u00a9 Esri"},
};
const BASEMAP_EARTH_RADIUS_M = 6378137;  // must match the map editor's lat/lon → metres projection
const BASEMAP_TILE_PX = 256;
const BASEMAP_MAX_VISIBLE_TILES = 400;
const BASEMAP_TILE_CACHE_SIZE = 1500;
const BASEMAP_MAX_FALLBACK_LEVELS = 4;
const BASEMAP_MAX_LATITUDE_DEG = 85.0511;

function createBasemap(onTileLoaded) {
  const tileCache = new Map();
  let redrawQueued = false;

  function queueRedraw() {
    if (redrawQueued) return;
    redrawQueued = true;
    requestAnimationFrame(() => { redrawQueued = false; onTileLoaded(); });
  }
  function tileImage(source, zoom, tileX, tileY, startLoading) {
    const key = `${source.url}|${zoom}|${tileX}|${tileY}`;
    if (tileCache.has(key)) return tileCache.get(key);
    if (!startLoading) return null;
    if (tileCache.size >= BASEMAP_TILE_CACHE_SIZE) tileCache.delete(tileCache.keys().next().value);
    const image = new Image();
    image.onload = queueRedraw;
    image.src = source.url.replace("{z}", zoom).replace("{x}", tileX).replace("{y}", tileY);
    tileCache.set(key, image);
    return image;
  }
  function metresPerDegree(origin) {
    const lat_m = BASEMAP_EARTH_RADIUS_M * Math.PI / 180;
    return {lat_m, lon_m: lat_m * Math.cos(origin.lat * Math.PI / 180)};
  }
  function tileLon(tileX, zoom) { return tileX / 2 ** zoom * 360 - 180; }
  function tileLat(tileY, zoom) { return Math.atan(Math.sinh(Math.PI * (1 - 2 * tileY / 2 ** zoom))) * 180 / Math.PI; }
  function tileIndexX(lon, zoom) { return Math.floor((lon + 180) / 360 * 2 ** zoom); }
  function tileIndexY(lat, zoom) {
    const latRad = Math.max(-BASEMAP_MAX_LATITUDE_DEG, Math.min(BASEMAP_MAX_LATITUDE_DEG, lat)) * Math.PI / 180;
    return Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * 2 ** zoom);
  }
  function loadedAncestor(source, zoom, tileX, tileY) {
    for (let level = 1; level <= Math.min(BASEMAP_MAX_FALLBACK_LEVELS, zoom); level++) {
      const image = tileImage(source, zoom - level, tileX >> level, tileY >> level, false);
      if (!image || !image.complete || !image.naturalWidth) continue;
      const cellCount = 2 ** level, cellPx = BASEMAP_TILE_PX / cellCount;
      return {image, sx: (tileX % cellCount) * cellPx, sy: (tileY % cellCount) * cellPx, size: cellPx};
    }
    return null;
  }

  // view: {toScreen, toWorld, width, height, scale (screen px per metre)}
  function draw(ctx, origin, sourceName, opacity, view) {
    const source = BASEMAP_SOURCES[sourceName];
    if (!source || !origin) return;
    const degree = metresPerDegree(origin);
    const toLatLon = ([x, y]) => [origin.lat + y / degree.lat_m, origin.lon + x / degree.lon_m];
    const toLocal = (lat, lon) => [(lon - origin.lon) * degree.lon_m, (lat - origin.lat) * degree.lat_m];
    const metresPerTilePxAtZoom0 = 2 * Math.PI * BASEMAP_EARTH_RADIUS_M * Math.cos(origin.lat * Math.PI / 180) / BASEMAP_TILE_PX;
    const zoom = Math.max(0, Math.min(source.max_zoom, Math.round(Math.log2(metresPerTilePxAtZoom0 * view.scale))));
    const [northLat, westLon] = toLatLon(view.toWorld(0, 0));
    const [southLat, eastLon] = toLatLon(view.toWorld(view.width, view.height));
    const lastTile = 2 ** zoom - 1;
    const x0 = Math.max(0, tileIndexX(westLon, zoom)), x1 = Math.min(lastTile, tileIndexX(eastLon, zoom));
    const y0 = Math.max(0, tileIndexY(northLat, zoom)), y1 = Math.min(lastTile, tileIndexY(southLat, zoom));
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > BASEMAP_MAX_VISIBLE_TILES) return;
    ctx.save();
    ctx.globalAlpha = opacity;
    for (let tileY = y0; tileY <= y1; tileY++) {
      for (let tileX = x0; tileX <= x1; tileX++) {
        const [left, top] = view.toScreen(...toLocal(tileLat(tileY, zoom), tileLon(tileX, zoom)));
        const [right, bottom] = view.toScreen(...toLocal(tileLat(tileY + 1, zoom), tileLon(tileX + 1, zoom)));
        const image = tileImage(source, zoom, tileX, tileY, true);
        if (image.complete && image.naturalWidth) {
          ctx.drawImage(image, left, top, right - left, bottom - top);
          continue;
        }
        const fallback = loadedAncestor(source, zoom, tileX, tileY);
        if (fallback) ctx.drawImage(fallback.image, fallback.sx, fallback.sy, fallback.size, fallback.size, left, top, right - left, bottom - top);
      }
    }
    ctx.globalAlpha = 1;
    const px = devicePixelRatio;
    ctx.font = `${11 * px}px system-ui, sans-serif`;
    const textWidth = ctx.measureText(source.attribution).width;
    ctx.fillStyle = "rgba(255,255,255,.8)";
    ctx.fillRect(view.width - textWidth - 10 * px, view.height - 18 * px, textWidth + 10 * px, 18 * px);
    ctx.fillStyle = "#333";
    ctx.fillText(source.attribution, view.width - textWidth - 5 * px, view.height - 5 * px);
    ctx.restore();
  }
  return {draw};
}
