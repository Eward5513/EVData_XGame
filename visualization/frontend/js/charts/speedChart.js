export function drawSpeedChart(speedPoints, trafficLightData, mode = 'single', vehicleIds = [], metricKeyParam = 'speed', getVehicleColorsFn = null) {
    const canvasContainer = document.getElementById('speedChartCanvas');
    if (!canvasContainer) {
        console.error('Canvas container not found');
        return;
    }

    if (!speedPoints || speedPoints.length === 0) {
        console.error('No speed data points to draw');
        canvasContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No speed data available to display</div>';
        return;
    }

    const canvas = document.createElement('canvas');
    const containerWidth = canvasContainer.offsetWidth || 800;
    const containerHeight = canvasContainer.offsetHeight || 400;
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    canvasContainer.innerHTML = '';
    canvasContainer.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const padding = 60;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;

    const collectionTimes = speedPoints.map(p => p.collectiontime);
    const metricKey = metricKeyParam;
    const values = speedPoints.map(p => {
        const n = Number(p[metricKey]);
        return Number.isFinite(n) ? n : 0;
    });

    const minTime = Math.min(...collectionTimes);
    const maxTime = Math.max(...collectionTimes);
    const timeToX = (timestamp) => {
        if (maxTime === minTime) return padding + chartWidth / 2;
        return padding + ((timestamp - minTime) / (maxTime - minTime)) * chartWidth;
    };

    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const valuePadding = Math.max(1, (maxVal - minVal) * 0.1);
    const valueMin = minVal - valuePadding;
    const valueMax = maxVal + valuePadding;
    const valueToY = (val) => {
        if (valueMax === valueMin) return padding + chartHeight / 2;
        const normalized = (val - valueMin) / (valueMax - valueMin);
        return padding + chartHeight * (1 - normalized);
    };

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (trafficLightData && trafficLightData.traffic_lights.length > 0) {
        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
        trafficLightData.traffic_lights.forEach(light => {
            const startTime = new Date(light.start_time);
            const endTime = new Date(light.end_time);
            if (!isNaN(startTime.getTime()) && !isNaN(endTime.getTime())) {
                const startTimestamp = startTime.getTime();
                const endTimestamp = endTime.getTime();
                const x1 = timeToX(startTimestamp);
                const x2 = timeToX(endTimestamp);
                if (!isNaN(x1) && !isNaN(x2)) {
                    ctx.fillRect(x1, padding, x2 - x1, chartHeight);
                }
            }
        });
    }

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.font = '12px Arial';
    ctx.fillStyle = '#666666';

    const speedSteps = 5;
    for (let i = 0; i <= speedSteps; i++) {
        const val = valueMin + (valueMax - valueMin) * (i / speedSteps);
        const y = valueToY(val);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(padding + chartWidth, y);
        ctx.stroke();
        ctx.fillText(val.toFixed(1), 5, y + 4);
    }

    const timeSteps = 6;
    for (let i = 0; i <= timeSteps; i++) {
        const timestamp = minTime + (maxTime - minTime) * (i / timeSteps);
        const x = timeToX(timestamp);
        const time = new Date(timestamp);
        const timeStr = time.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, padding + chartHeight);
        ctx.stroke();
        ctx.save();
        ctx.translate(x, canvas.height - 10);
        ctx.rotate(-Math.PI / 6);
        ctx.fillText(timeStr, -20, 0);
        ctx.restore();
    }

    const vehicleGroups = {};
    if (mode === 'single') {
        const vehicleId = speedPoints[0]?.vehicle_id || 1;
        vehicleGroups[vehicleId] = speedPoints;
    } else {
        speedPoints.forEach(point => {
            if (!vehicleGroups[point.vehicle_id]) {
                vehicleGroups[point.vehicle_id] = [];
            }
            vehicleGroups[point.vehicle_id].push(point);
        });
    }

    const defaultGetVehicleColors = (count) => {
        const arr = [];
        const base = [
            Cesium.Color.CYAN,
            Cesium.Color.ORANGE,
            Cesium.Color.LIME,
            Cesium.Color.MAGENTA,
            Cesium.Color.YELLOW,
            Cesium.Color.LIGHTBLUE,
            Cesium.Color.LIGHTGREEN,
            Cesium.Color.PINK,
            Cesium.Color.LIGHTCYAN,
            Cesium.Color.LIGHTGRAY
        ];
        for (let i = 0; i < count; i++) {
            if (i < base.length) arr.push(base[i]);
            else arr.push(Cesium.Color.fromHsl(((i * 137.5) % 360) / 360, 0.7, 0.6));
        }
        return arr;
    };

    const vehicleColors = (getVehicleColorsFn || defaultGetVehicleColors)(Object.keys(vehicleGroups).length);
    let colorIndex = 0;

    Object.keys(vehicleGroups).forEach(vehicleId => {
        const vehicleData = vehicleGroups[vehicleId];
        const vehicleColor = mode === 'single' ? { red: 0.30, green: 0.69, blue: 0.31 } : vehicleColors[colorIndex];
        const colorStr = `rgb(${Math.floor(vehicleColor.red * 255)}, ${Math.floor(vehicleColor.green * 255)}, ${Math.floor(vehicleColor.blue * 255)})`;

        vehicleData.sort((a, b) => a.collectiontime - b.collectiontime);

        ctx.strokeStyle = colorStr;
        ctx.lineWidth = mode === 'single' ? 2 : 1.5;
        ctx.beginPath();

        let validPoints = 0;
        vehicleData.forEach((point) => {
            const x = timeToX(point.collectiontime);
            const y = valueToY(point[metricKey]);
            if (isNaN(x) || isNaN(y)) return;
            const clampedX = Math.max(padding, Math.min(canvas.width - padding, x));
            const clampedY = Math.max(padding, Math.min(canvas.height - padding, y));
            validPoints++;
            if (validPoints === 1) ctx.moveTo(clampedX, clampedY);
            else ctx.lineTo(clampedX, clampedY);
        });
        ctx.stroke();

        ctx.fillStyle = colorStr;
        vehicleData.forEach((point) => {
            const x = timeToX(point.collectiontime);
            const y = valueToY(point[metricKey]);
            if (isNaN(x) || isNaN(y)) return;
            const clampedX = Math.max(padding, Math.min(canvas.width - padding, x));
            const clampedY = Math.max(padding, Math.min(canvas.height - padding, y));
            ctx.beginPath();
            ctx.arc(clampedX, clampedY, mode === 'single' ? 4 : 3, 0, 2 * Math.PI);
            ctx.fill();
        });

        colorIndex++;
    });

    if (trafficLightData && trafficLightData.traffic_lights.length > 0) {
        ctx.fillStyle = '#dc3545';
        ctx.font = '10px Arial';
        trafficLightData.traffic_lights.forEach((light, index) => {
            const startTime = new Date(light.start_time);
            if (!isNaN(startTime.getTime())) {
                const x = timeToX(startTime.getTime());
                if (!isNaN(x)) {
                    const y = padding + 20 + (index % 3) * 15;
                    ctx.fillText(`${light.phase_id}`, x, y);
                }
            }
        });
    }

    ctx.fillStyle = '#333333';
    ctx.font = '14px Arial';
    ctx.fillText(`${metricKey}`, 10, 30);
    ctx.fillText('Time', canvas.width / 2 - 20, canvas.height - 10);
    ctx.strokeStyle = '#cccccc';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, chartWidth, chartHeight);
}













