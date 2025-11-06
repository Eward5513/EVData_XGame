export function drawAnalysisCenterOverlay(overlayDataSource, analysisCenters, centerChangeRadiusMeters) {
    if (!overlayDataSource) return;
    overlayDataSource.entities.removeAll();

    const r = centerChangeRadiusMeters;

    Object.entries(analysisCenters).forEach(([roadId, center]) => {
        const lon = center.lon;
        const lat = center.lat;
        const centerPos = Cesium.Cartesian3.fromDegrees(lon, lat);

        const pointColor = roadId === 'A0008' ? Cesium.Color.MAGENTA : Cesium.Color.CYAN;
        const outlineColor = roadId === 'A0008' ? Cesium.Color.YELLOW : Cesium.Color.RED;
        const fillColor = (roadId === 'A0008' ? Cesium.Color.MAGENTA : Cesium.Color.CYAN).withAlpha(0.15);

        overlayDataSource.entities.add({
            id: `analysis_center_point_${roadId}`,
            position: centerPos,
            point: {
                pixelSize: 14,
                color: pointColor,
                outlineColor: outlineColor,
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
            },
            description: `<h3>${roadId} Direction Analysis Center</h3><p>Lon: ${lon.toFixed(6)}, Lat: ${lat.toFixed(6)}</p>`
        });

        overlayDataSource.entities.add({
            id: `analysis_center_radius_${roadId}`,
            position: centerPos,
            ellipse: {
                semiMajorAxis: r,
                semiMinorAxis: r,
                material: fillColor,
                outline: true,
                outlineColor: outlineColor,
                outlineWidth: 2
            },
            description: `<h3>${roadId} Near-Center Radius</h3><p>Radius: ${r.toFixed(1)} m</p>`
        });
    });
}









