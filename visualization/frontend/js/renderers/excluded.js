export function renderExcludedTrajectories(
    excludedDataSource,
    viewer,
    data,
    getSpeedColor,
    getVehicleColors,
    blendColors,
    createPointDescription
) {
    if (!excludedDataSource || !viewer) return;
    excludedDataSource.entities.removeAll();
    if (!Array.isArray(data) || data.length === 0) return;

    const groups = {};
    data.forEach((p) => {
        const segId = (typeof p.seg_id === 'number') ? p.seg_id : 0;
        const key = `${p.vehicle_id}_${p.date}_${segId}`;
        if (!groups[key]) groups[key] = [];
        groups[key].push(p);
    });

    const trajectoryKeys = Object.keys(groups);
    const colors = getVehicleColors(trajectoryKeys.length);

    trajectoryKeys.forEach((trajectoryKey, trajectoryIndex) => {
        const pts = groups[trajectoryKey];
        pts.sort((a,b)=> (a.collectiontime||0)-(b.collectiontime||0));
        const positions = [];
        const trajectoryColor = colors[trajectoryIndex];
        const trajectoryDate = pts.length > 0 ? pts[0].date : null;

        pts.forEach((point, index) => {
            const position = Cesium.Cartesian3.fromDegrees(point.longitude, point.latitude, 0);
            positions.push(position);
            let pointColor = getSpeedColor(point.speed);
            if (trajectoryKeys.length > 1) {
                pointColor = blendColors(trajectoryColor, pointColor);
            }
            excludedDataSource.entities.add({
                position: position,
                point: {
                    pixelSize: 8,
                    color: pointColor,
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: 2,
                    heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
                },
                description: createPointDescription(point, index, trajectoryKeys.length > 1, trajectoryDate)
            });
        });

        if (positions.length > 1) {
            excludedDataSource.entities.add({
                polyline: {
                    positions: positions,
                    width: trajectoryKeys.length > 1 ? 4 : 3,
                    material: trajectoryColor.withAlpha(0.8),
                    clampToGround: true
                },
                description: `Vehicle ${pts[0].vehicle_id} - ${pts[0].date} - seg ${pts[0].seg_id} (${pts.length} points)`
            });
        }
    });

    viewer.flyTo(excludedDataSource);
}






