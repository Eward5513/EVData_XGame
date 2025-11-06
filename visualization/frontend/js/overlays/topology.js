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
