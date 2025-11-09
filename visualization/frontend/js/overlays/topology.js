// Topology display code removed; using infer overlays as unified entry

export function drawWestStoplines(topologyDataSource, stoplinesByRoad) {
    if (!topologyDataSource || !stoplinesByRoad || typeof stoplinesByRoad !== 'object') return;
    try {
        Object.keys(stoplinesByRoad).forEach((roadId) => {
            const obj = stoplinesByRoad[roadId] || {};
            const seg = obj.stopline_segment;
            const sStop = (obj.s_stop_m !== undefined && obj.s_stop_m !== null) ? Number(obj.s_stop_m) : null;
            if (!Array.isArray(seg) || seg.length < 2) return;
            const [p1, p2] = seg;
            if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) return;
            const color = (roadId === 'A0008') ? Cesium.Color.YELLOW.withAlpha(0.95) : Cesium.Color.ORANGE.withAlpha(0.95);
            topologyDataSource.entities.add({
                id: `west_stopline_${roadId}`,
                polyline: {
                    positions: [
                        Cesium.Cartesian3.fromDegrees(Number(p1[0]), Number(p1[1])),
                        Cesium.Cartesian3.fromDegrees(Number(p2[0]), Number(p2[1]))
                    ],
                    width: 6,
                    material: color,
                    clampToGround: true
                },
                description: `<h3>${roadId} West Stopline</h3>${sStop!==null && Number.isFinite(sStop) ? `<p>s_stop_m: ${sStop.toFixed(2)} m</p>` : ''}`
            });
        });
    } catch (e) {
        console.warn('Failed to render west stoplines:', e);
    }
}

export function drawEastStoplines(topologyDataSource, stoplinesByRoad) {
    if (!topologyDataSource || !stoplinesByRoad || typeof stoplinesByRoad !== 'object') return;
    try {
        Object.keys(stoplinesByRoad).forEach((roadId) => {
            const obj = stoplinesByRoad[roadId] || {};
            const seg = obj.stopline_segment;
            const sStop = (obj.s_stop_m !== undefined && obj.s_stop_m !== null) ? Number(obj.s_stop_m) : null;
            if (!Array.isArray(seg) || seg.length < 2) return;
            const [p1, p2] = seg;
            if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) return;
            const color = (roadId === 'A0008') ? Cesium.Color.LIME.withAlpha(0.95) : Cesium.Color.DODGERBLUE.withAlpha(0.95);
            topologyDataSource.entities.add({
                id: `east_stopline_${roadId}`,
                polyline: {
                    positions: [
                        Cesium.Cartesian3.fromDegrees(Number(p1[0]), Number(p1[1])),
                        Cesium.Cartesian3.fromDegrees(Number(p2[0]), Number(p2[1]))
                    ],
                    width: 6,
                    material: color,
                    clampToGround: true
                },
                description: `<h3>${roadId} East Stopline</h3>${sStop!==null && Number.isFinite(sStop) ? `<p>s_stop_m: ${sStop.toFixed(2)} m</p>` : ''}`
            });
        });
    } catch (e) {
        console.warn('Failed to render east stoplines:', e);
    }
}

export function drawStoplinesAll(topologyDataSource, stoplinesNestedByRoad) {
    if (!topologyDataSource || !stoplinesNestedByRoad || typeof stoplinesNestedByRoad !== 'object') return;
    try {
        const colorFor = (roadId, dir) => {
            if (dir === 'west') return (roadId === 'A0008') ? Cesium.Color.YELLOW.withAlpha(0.95) : Cesium.Color.ORANGE.withAlpha(0.95);
            if (dir === 'east') return (roadId === 'A0008') ? Cesium.Color.LIME.withAlpha(0.95) : Cesium.Color.DODGERBLUE.withAlpha(0.95);
            if (dir === 'south') return Cesium.Color.RED.withAlpha(0.95);
            if (dir === 'north') return Cesium.Color.SKYBLUE.withAlpha(0.95);
            return Cesium.Color.WHITE.withAlpha(0.9);
        };

        const meanLat = (points) => {
            const vals = points.map(p => Number(p[1])).filter(v => Number.isFinite(v));
            if (!vals.length) return 0.0;
            return vals.reduce((a,b)=>a+b,0) / vals.length;
        };

        const intersectLonLat = (segA, segB, cosLat) => {
            try {
                const [a1, a2] = segA;
                const [b1, b2] = segB;
                const ax1 = Number(a1[0]) * cosLat, ay1 = Number(a1[1]);
                const ax2 = Number(a2[0]) * cosLat, ay2 = Number(a2[1]);
                const bx1 = Number(b1[0]) * cosLat, by1 = Number(b1[1]);
                const bx2 = Number(b2[0]) * cosLat, by2 = Number(b2[1]);

                const dax = ax2 - ax1, day = ay2 - ay1;
                const dbx = bx2 - bx1, dby = by2 - by1;
                const denom = dax * dby - day * dbx;
                if (Math.abs(denom) < 1e-12) return null; // parallel or near-parallel
                const dx = bx1 - ax1, dy = by1 - ay1;
                const t = (dx * dby - dy * dbx) / denom;
                const ix = ax1 + t * dax;
                const iy = ay1 + t * day;
                const lon = ix / cosLat;
                const lat = iy;
                if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
                return [lon, lat];
            } catch (_) {
                return null;
            }
        };

        Object.keys(stoplinesNestedByRoad).forEach((roadId) => {
            const obj = stoplinesNestedByRoad[roadId] || {};
            const segByDir = {};
            ['west','east','south','north'].forEach((dir) => {
                try {
                    const item = obj[dir];
                    if (!item || !Array.isArray(item.stopline_segment)) return;
                    const seg = item.stopline_segment;
                    if (!Array.isArray(seg) || seg.length < 2) return;
                    const [p1, p2] = seg;
                    if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) return;
                    segByDir[dir] = [p1, p2];
                    const color = colorFor(roadId, dir);
                    const lineId = `stopline_${roadId}_${dir}`;
                    if (topologyDataSource.entities.getById(lineId)) {
                        topologyDataSource.entities.removeById(lineId);
                    }
                    topologyDataSource.entities.add({
                        id: lineId,
                        polyline: {
                            positions: [
                                Cesium.Cartesian3.fromDegrees(Number(p1[0]), Number(p1[1])),
                                Cesium.Cartesian3.fromDegrees(Number(p2[0]), Number(p2[1]))
                            ],
                            width: 6,
                            material: color,
                            clampToGround: true
                        },
                        description: `<h3>${roadId} ${dir.toUpperCase()} Stopline</h3>`
                    });
                } catch (_) { /* skip malformed one */ }
            });

            // Draw rectangle from four stopline intersections if all four present
            try {
                if (segByDir.west && segByDir.east && segByDir.south && segByDir.north) {
                    const allPts = [segByDir.west[0], segByDir.west[1], segByDir.east[0], segByDir.east[1], segByDir.south[0], segByDir.south[1], segByDir.north[0], segByDir.north[1]];
                    const lat0 = meanLat(allPts);
                    const cosLat = Math.cos(lat0 * Math.PI / 180.0) || 1.0;
                    const NW = intersectLonLat(segByDir.west, segByDir.north, cosLat);
                    const SW = intersectLonLat(segByDir.west, segByDir.south, cosLat);
                    const NE = intersectLonLat(segByDir.east, segByDir.north, cosLat);
                    const SE = intersectLonLat(segByDir.east, segByDir.south, cosLat);
                    if (NW && NE && SE && SW) {
                        const degs = [
                            NW[0], NW[1],
                            NE[0], NE[1],
                            SE[0], SE[1],
                            SW[0], SW[1]
                        ];
                        const positions = Cesium.Cartesian3.fromDegreesArray(degs);
                        const rectId = `stopline_rect_${roadId}`;
                        if (topologyDataSource.entities.getById(rectId)) {
                            topologyDataSource.entities.removeById(rectId);
                        }
                        topologyDataSource.entities.add({
                            id: rectId,
                            polygon: {
                                hierarchy: new Cesium.PolygonHierarchy(positions),
                                material: Cesium.Color.WHITE.withAlpha(0.15),
                                outline: true,
                                outlineColor: Cesium.Color.WHITE.withAlpha(0.9)
                            },
                            description: `<h3>${roadId} Stopline Box</h3>`
                        });

                        // Corner markers
                        const corners = [
                            { id: `stopline_corner_${roadId}_NW`, pt: NW },
                            { id: `stopline_corner_${roadId}_NE`, pt: NE },
                            { id: `stopline_corner_${roadId}_SE`, pt: SE },
                            { id: `stopline_corner_${roadId}_SW`, pt: SW }
                        ];
                        corners.forEach(({ id, pt }) => {
                            if (topologyDataSource.entities.getById(id)) {
                                topologyDataSource.entities.removeById(id);
                            }
                            topologyDataSource.entities.add({
                                id,
                                position: Cesium.Cartesian3.fromDegrees(pt[0], pt[1]),
                                point: {
                                    pixelSize: 8,
                                    color: Cesium.Color.WHITE,
                                    outlineColor: Cesium.Color.BLACK,
                                    outlineWidth: 2
                                }
                            });
                        });
                    }
                }
            } catch (e) {
                // ignore rectangle if intersection fails
            }
        });
    } catch (e) {
        console.warn('Failed to render stoplines (all directions):', e);
    }
}
