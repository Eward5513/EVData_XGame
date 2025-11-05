import { createArrowCanvas } from '../utils/geo.js';

export function displayTopology(geojson, summary, topologyDataSource, viewer) {
    if (!topologyDataSource) return;
    topologyDataSource.entities.removeAll();

    if (!geojson || !geojson.features) {
        console.warn('No features in topology data');
        return;
    }

    console.log(`Displaying ${geojson.features.length} topology features`);

    geojson.features.forEach(feature => {
        const props = feature.properties || {};
        const geom = feature.geometry;

        if (props.kind === 'center') {
            const [lon, lat] = geom.coordinates;
            topologyDataSource.entities.add({
                position: Cesium.Cartesian3.fromDegrees(lon, lat),
                point: {
                    pixelSize: 15,
                    color: Cesium.Color.RED,
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: 2
                },
                label: {
                    text: 'Intersection Center',
                    font: '14px sans-serif',
                    fillColor: Cesium.Color.WHITE,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                    pixelOffset: new Cesium.Cartesian2(0, -20)
                },
                description: `<h3>Intersection Center</h3><p>Coordinates: ${lon.toFixed(6)}, ${lat.toFixed(6)}</p>`
            });
        } else if (props.kind === 'arm_node') {
            const [lon, lat] = geom.coordinates;
            const armId = props.id;
            const angleRad = props.angle_rad;
            const angleDeg = (angleRad * 180 / Math.PI).toFixed(1);

            topologyDataSource.entities.add({
                position: Cesium.Cartesian3.fromDegrees(lon, lat),
                point: {
                    pixelSize: 12,
                    color: Cesium.Color.BLUE,
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: 2
                },
                label: {
                    text: `Arm ${armId}`,
                    font: '12px sans-serif',
                    fillColor: Cesium.Color.YELLOW,
                    outlineColor: Cesium.Color.BLACK,
                    outlineWidth: 2,
                    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                    pixelOffset: new Cesium.Cartesian2(0, -15)
                },
                description: `<h3>Arm ${armId}</h3><p>Angle: ${angleDeg}°</p><p>Coordinates: ${lon.toFixed(6)}, ${lat.toFixed(6)}</p>`
            });
        } else if (props.kind === 'movement') {
            const coords = geom.coordinates;
            const positions = coords.map(coord => Cesium.Cartesian3.fromDegrees(coord[0], coord[1]));
            const weight = props.weight || 1;

            let color = Cesium.Color.GREEN;
            if (weight > 15) {
                color = Cesium.Color.RED;
            } else if (weight > 10) {
                color = Cesium.Color.ORANGE;
            } else if (weight > 5) {
                color = Cesium.Color.YELLOW;
            }

            topologyDataSource.entities.add({
                polyline: {
                    positions: positions,
                    width: Math.min(2 + weight * 0.5, 10),
                    material: color.withAlpha(0.8),
                    clampToGround: true
                },
                description: `<h3>Movement Edge</h3><p>From Arm ${props.u} to Arm ${props.v}</p><p>Traffic Volume: ${weight} vehicles</p>`
            });

            if (coords.length >= 2) {
                const lastCoord = coords[coords.length - 1];
                const secondLastCoord = coords[coords.length - 2];

                const dx = lastCoord[0] - secondLastCoord[0];
                const dy = lastCoord[1] - secondLastCoord[1];
                const heading = Math.atan2(dx, dy);

                topologyDataSource.entities.add({
                    position: Cesium.Cartesian3.fromDegrees(lastCoord[0], lastCoord[1]),
                    billboard: {
                        image: createArrowCanvas(color),
                        scale: 0.5,
                        rotation: -heading,
                        alignedAxis: Cesium.Cartesian3.UNIT_Z
                    }
                });
            }
        }
    });

    console.log('Topology display completed');

    if (viewer && geojson.features && geojson.features.length > 0) {
        const centerFeature = geojson.features.find(f => f.properties.kind === 'center');
        if (centerFeature && centerFeature.geometry.coordinates) {
            const [lon, lat] = centerFeature.geometry.coordinates;
            viewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(lon, lat, 500),
                duration: 2.0
            });
        }
    }
}








