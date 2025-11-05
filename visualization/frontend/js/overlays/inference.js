export function drawIntersectionInference(topologyDataSource, roadId, inference, offsetLonLatFn) {
    if (!inference || !inference.center_point || !topologyDataSource) return;
    const cp = inference.center_point;
    const centerLon = Number(cp.lon);
    const centerLat = Number(cp.lat);
    const centerX = Number(cp.x_m);
    const centerY = Number(cp.y_m);
    const centerXY = [centerX, centerY];

    const colorAxis = (roadId === 'A0008') ? Cesium.Color.MAGENTA : Cesium.Color.CYAN;
    const colorStop = (roadId === 'A0008') ? Cesium.Color.YELLOW : Cesium.Color.ORANGE;
    const colorCenter = (roadId === 'A0008') ? Cesium.Color.MAGENTA : Cesium.Color.CYAN;

    topologyDataSource.entities.add({
        id: `intersection_center_${roadId}`,
        position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat),
        point: {
            pixelSize: 12,
            color: colorCenter,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2
        },
        label: {
            text: `${roadId} Center`,
            font: '14px sans-serif',
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -20)
        }
    });

    const L_vis = 800.0;
    const Ls = 20.0;

    (inference.approaches || []).forEach(app => {
        try {
            const axis = app.axis || {};
            const angleDeg = Number(axis.angle_deg) || 0.0;
            const rho = Number(axis.rho_m) || 0.0;
            const n = axis.n || [0, 1];
            const nDotCenter = n[0] * centerXY[0] + n[1] * centerXY[1];
            const alpha = rho - nDotCenter;

            const nAngle = (angleDeg + 90.0) % 360.0;
            const tAngle = angleDeg;

            const axisPolyline = app.axis_polyline && Array.isArray(app.axis_polyline.coordinates) ? app.axis_polyline.coordinates : null;
            if (axisPolyline && axisPolyline.length >= 2) {
                const positions = axisPolyline.map(([lo, la]) => Cesium.Cartesian3.fromDegrees(Number(lo), Number(la)));
                topologyDataSource.entities.add({
                    id: `axis_${roadId}_${app.cluster_label}`,
                    polyline: {
                        positions,
                        width: 6,
                        material: new Cesium.PolylineDashMaterialProperty({
                            color: colorAxis.withAlpha(0.95),
                            dashLength: 16.0
                        }),
                        clampToGround: true
                    },
                    description: `<h3>Axis ${app.label_hint || app.cluster_label}</h3><p>Angle: ${angleDeg.toFixed(1)}° (piecewise)</p>`
                });

                axisPolyline.forEach(([lo, la], idx) => {
                    const lonV = Number(lo);
                    const latV = Number(la);
                    topologyDataSource.entities.add({
                        position: Cesium.Cartesian3.fromDegrees(lonV, latV),
                        point: {
                            pixelSize: 8,
                            color: colorAxis,
                            outlineColor: Cesium.Color.WHITE,
                            outlineWidth: 2
                        },
                        description: `Axis vertex #${idx + 1}`
                    });
                });
            } else {
                const [axisBaseLon, axisBaseLat] = offsetLonLatFn(centerLon, centerLat, alpha, nAngle);
                const [lon1, lat1] = offsetLonLatFn(axisBaseLon, axisBaseLat, -L_vis / 2, tAngle);
                const [lon2, lat2] = offsetLonLatFn(axisBaseLon, axisBaseLat, +L_vis / 2, tAngle);

                topologyDataSource.entities.add({
                    id: `axis_${roadId}_${app.cluster_label}`,
                    polyline: {
                        positions: [
                            Cesium.Cartesian3.fromDegrees(lon1, lat1),
                            Cesium.Cartesian3.fromDegrees(lon2, lat2)
                        ],
                        width: 6,
                        material: new Cesium.PolylineDashMaterialProperty({
                            color: colorAxis.withAlpha(0.95),
                            dashLength: 16.0
                        }),
                        clampToGround: true
                    },
                    description: `<h3>Axis ${app.label_hint || app.cluster_label}</h3><p>Angle: ${angleDeg.toFixed(1)}°</p>`
                });

                const endpoints = [
                    [lon1, lat1],
                    [lon2, lat2]
                ];
                endpoints.forEach(([lo, la], idx) => {
                    topologyDataSource.entities.add({
                        position: Cesium.Cartesian3.fromDegrees(lo, la),
                        point: {
                            pixelSize: 8,
                            color: colorAxis,
                            outlineColor: Cesium.Color.WHITE,
                            outlineWidth: 2
                        },
                        description: `Axis endpoint #${idx + 1}`
                    });
                });
            }

            const sStop = app.stopline && (app.stopline.s_stop_m_from_center !== null && app.stopline.s_stop_m_from_center !== undefined)
                ? Number(app.stopline.s_stop_m_from_center) : null;
            if (sStop !== null && Number.isFinite(sStop)) {
                const [sLon, sLat] = offsetLonLatFn(centerLon, centerLat, sStop, tAngle);
                const [q1Lon, q1Lat] = offsetLonLatFn(sLon, sLat, -Ls / 2, nAngle);
                const [q2Lon, q2Lat] = offsetLonLatFn(sLon, sLat, +Ls / 2, nAngle);
                topologyDataSource.entities.add({
                    id: `stopline_${roadId}_${app.cluster_label}`,
                    polyline: {
                        positions: [
                            Cesium.Cartesian3.fromDegrees(q1Lon, q1Lat),
                            Cesium.Cartesian3.fromDegrees(q2Lon, q2Lat)
                        ],
                        width: 5,
                        material: colorStop.withAlpha(0.9),
                        clampToGround: true
                    },
                    description: `<h3>Stop Line ${app.label_hint || app.cluster_label}</h3><p>s from center: ${sStop.toFixed(2)} m</p>`
                });
            }
        } catch (e) {
            console.warn('Failed to render approach overlay:', e);
        }
    });

    // Render straight lanes per side if available
    try {
        const lanesStraight = inference.lanes_straight || {};
        const laneColor = (roadId === 'A0008') ? Cesium.Color.LIME : Cesium.Color.GREEN;
        const laneVis = 600.0;
        ['W','E','S','N'].forEach(side => {
            const sideObj = lanesStraight[side];
            if (!sideObj || !Array.isArray(sideObj.lanes)) return;
            sideObj.lanes.forEach((ln, idx) => {
                try {
                    const angleDeg = Number(ln.angle_deg) || 0.0;
                    const rho = Number(ln.rho_m) || 0.0;
                    const n = ln.n || [0, 1];
                    const nDotCenter = n[0] * centerXY[0] + n[1] * centerXY[1];
                    const alpha = rho - nDotCenter;
                    const nAngle = (angleDeg + 90.0) % 360.0;
                    const tAngle = angleDeg;
                    const [baseLon, baseLat] = offsetLonLatFn(centerLon, centerLat, alpha, nAngle);
                    const [lon1, lat1] = offsetLonLatFn(baseLon, baseLat, -laneVis / 2, tAngle);
                    const [lon2, lat2] = offsetLonLatFn(baseLon, baseLat, +laneVis / 2, tAngle);
                    topologyDataSource.entities.add({
                        id: `lane_${roadId}_${side}_${idx}`,
                        polyline: {
                            positions: [
                                Cesium.Cartesian3.fromDegrees(lon1, lat1),
                                Cesium.Cartesian3.fromDegrees(lon2, lat2)
                            ],
                            width: 4,
                            material: laneColor.withAlpha(0.85),
                            clampToGround: true
                        },
                        description: `<h3>Straight Lane ${side}-${idx+1}</h3><p>Angle: ${angleDeg.toFixed(1)}°</p>`
                    });
                } catch (e) {
                    // skip one lane on error
                }
            });
        });
    } catch (e) {
        console.warn('Failed to render straight lanes:', e);
    }

    // Render turn polylines if available
    try {
        const lanesTurn = inference.lanes_turn || {};
        const leftColor = Cesium.Color.DODGERBLUE.withAlpha(0.9);
        const rightColor = Cesium.Color.HOTPINK.withAlpha(0.9);
        Object.entries(lanesTurn).forEach(([side, dirs]) => {
            if (dirs && dirs.left && Array.isArray(dirs.left.coordinates)) {
                const coords = dirs.left.coordinates.map(([lo, la]) => Cesium.Cartesian3.fromDegrees(Number(lo), Number(la)));
                if (coords.length >= 2) {
                    topologyDataSource.entities.add({
                        id: `turn_left_${roadId}_${side}`,
                        polyline: {
                            positions: coords,
                            width: 5,
                            material: leftColor,
                            clampToGround: true
                        },
                        description: `<h3>Left Turn (${side})</h3>`
                    });
                }
            }
            if (dirs && dirs.right && Array.isArray(dirs.right.coordinates)) {
                const coords = dirs.right.coordinates.map(([lo, la]) => Cesium.Cartesian3.fromDegrees(Number(lo), Number(la)));
                if (coords.length >= 2) {
                    topologyDataSource.entities.add({
                        id: `turn_right_${roadId}_${side}`,
                        polyline: {
                            positions: coords,
                            width: 5,
                            material: rightColor,
                            clampToGround: true
                        },
                        description: `<h3>Right Turn (${side})</h3>`
                    });
                }
            }
        });
    } catch (e) {
        console.warn('Failed to render turn polylines:', e);
    }
}







