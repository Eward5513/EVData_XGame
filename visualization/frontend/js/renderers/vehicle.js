export function renderVehicleTrajectories(
    dataSource,
    viewer,
    data,
    getSpeedColor,
    getVehicleColors,
    blendColors,
    createPointDescription,
    pointsOnly = false
) {
    if (!dataSource || !viewer) return;
    dataSource.entities.removeAll();
    if (!Array.isArray(data) || data.length === 0) return;

    const vehicleGroups = {};
    data.forEach(point => {
        const segId = (typeof point.seg_id === 'number') ? point.seg_id : 0;
        const trajectoryKey = `${point.vehicle_id}_${point.date}_${segId}`;
        if (!vehicleGroups[trajectoryKey]) {
            vehicleGroups[trajectoryKey] = {
                vehicle_id: point.vehicle_id,
                date: point.date,
                seg_id: segId,
                points: []
            };
        }
        vehicleGroups[trajectoryKey].points.push(point);
    });

    const trajectoryKeys = Object.keys(vehicleGroups);
    const colors = getVehicleColors(trajectoryKeys.length);

    trajectoryKeys.forEach((trajectoryKey, trajectoryIndex) => {
        const trajectoryInfo = vehicleGroups[trajectoryKey];
        const trajectoryData = trajectoryInfo.points;
        const trajectoryColor = colors[trajectoryIndex];
        const positions = [];

        trajectoryData.forEach((point, index) => {
            const position = Cesium.Cartesian3.fromDegrees(
                point.longitude,
                point.latitude,
                0
            );

            positions.push(position);

            let pointColor = getSpeedColor(point.speed);
            if (trajectoryKeys.length > 1) {
                pointColor = blendColors(trajectoryColor, pointColor);
            }

            dataSource.entities.add({
                position: position,
                point: {
                    pixelSize: 8,
                    color: pointColor,
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: point.end_time ? 2 : 0,
                    heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
                },
                description: createPointDescription(point, index, trajectoryKeys.length > 1, trajectoryInfo.date)
            });
        });

        if (!pointsOnly && positions.length > 1) {
            dataSource.entities.add({
                polyline: {
                    positions: positions,
                    width: trajectoryKeys.length > 1 ? 4 : 3,
                    material: trajectoryColor.withAlpha(0.8),
                    clampToGround: true
                },
                description: `Vehicle ${trajectoryInfo.vehicle_id} - ${trajectoryInfo.date} - seg ${trajectoryInfo.seg_id} (${trajectoryData.length} points)`
            });
        }
    });

    viewer.flyTo(dataSource);
}






