export function getSpeedColor(speed) {
    if (speed < 10) {
        return Cesium.Color.GREEN;
    } else if (speed < 30) {
        return Cesium.Color.YELLOW;
    } else if (speed < 50) {
        return Cesium.Color.ORANGE;
    } else {
        return Cesium.Color.RED;
    }
}

export function getVehicleColors(vehicleCount) {
    const baseColors = [
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

    const colors = [];
    for (let i = 0; i < vehicleCount; i++) {
        if (i < baseColors.length) {
            colors.push(baseColors[i]);
        } else {
            const hue = (i * 137.5) % 360;
            colors.push(Cesium.Color.fromHsl(hue / 360, 0.7, 0.6));
        }
    }
    return colors;
}

export function blendColors(vehicleColor, speedColor, ratio = 0.7) {
    return new Cesium.Color(
        vehicleColor.red * ratio + speedColor.red * (1 - ratio),
        vehicleColor.green * ratio + speedColor.green * (1 - ratio),
        vehicleColor.blue * ratio + speedColor.blue * (1 - ratio),
        1.0
    );
}









