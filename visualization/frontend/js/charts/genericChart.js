export function drawGenericMetricChart(containerEl, points, metricKey, title = '') {
    if (!containerEl) return;
    const width = containerEl.clientWidth || 800;
    const height = 320;
    const padding = 50;
    const canvasId = 'genericMetricCanvas';
    containerEl.innerHTML = `<div style="margin-bottom:6px;font-weight:bold;">${title} - ${metricKey}</div><canvas id="${canvasId}" width="${width}" height="${height}" style="border:1px solid #ccc;border-radius:4px;"></canvas>`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const times = points.map(p => p.collectiontime);
    const values = points.map(p => {
        const v = p[metricKey];
        if (metricKey === 'gearnum' || metricKey === 'havebrake' || metricKey === 'havedriver') {
            const n = Number(v);
            return Number.isFinite(n) ? n : 0;
        }
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
    });
    if (times.length === 0) return;
    const minT = Math.min(...times);
    const maxT = Math.max(...times);
    const minV = Math.min(...values);
    const maxV = Math.max(...values);

    const xMap = (t) => {
        if (maxT === minT) return padding + (width - 2 * padding) / 2;
        return padding + (t - minT) / (maxT - minT) * (width - 2 * padding);
    };
    const yMap = (v) => {
        if (maxV === minV) return padding + (height - 2 * padding) / 2;
        const norm = (v - minV) / (maxV - minV);
        return padding + (height - 2 * padding) * (1 - norm);
    };

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.fillStyle = '#000';
    ctx.font = '14px sans-serif';
    ctx.fillText(`${metricKey} over time`, padding, padding - 12);

    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
        const x = xMap(p.collectiontime);
        const y = yMap(values[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#007bff';
    points.forEach((p, i) => {
        const x = xMap(p.collectiontime);
        const y = yMap(values[i]);
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
    });
}









