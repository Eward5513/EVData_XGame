export function offsetLonLat(lon, lat, distanceMeters, angleDeg) {
    const R = 6378137.0;
    const rad = Math.PI / 180.0;
    const deg = 180.0 / Math.PI;
    const theta = angleDeg * rad; // 0=E, CCW positive
    const dx = distanceMeters * Math.cos(theta);
    const dy = distanceMeters * Math.sin(theta);
    const latRad = lat * rad;
    const dLon = (dx / (R * Math.cos(latRad))) * deg;
    const dLat = (dy / R) * deg;
    return [lon + dLon, lat + dLat];
}

export function createArrowCanvas(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = color.toCssColorString();
    ctx.beginPath();
    ctx.moveTo(16, 4);
    ctx.lineTo(28, 16);
    ctx.lineTo(20, 16);
    ctx.lineTo(20, 28);
    ctx.lineTo(12, 28);
    ctx.lineTo(12, 16);
    ctx.lineTo(4, 16);
    ctx.closePath();
    ctx.fill();

    return canvas;
}









