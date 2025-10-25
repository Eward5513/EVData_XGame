// Geographic Data Visualization Application Main File

class GeoVisualization {
    constructor() {
        this.viewer = null;
        this.dataSource = null;
        this.topologyDataSource = null;
        this.overlayDataSource = null;
        this.currentVehicleData = [];
        this.currentTrafficLightData = null;
        this.currentSpeedData = null;
        this.currentSpeedTrafficLights = null;
        this.currentTopologyData = null;
        this.apiBaseUrl = 'http://127.0.0.1:5000/api';
        this.currentSelectedMetric = 'speed';
        this.currentSelectedSegId = null;
        
		// Analysis centers and radius (A0003 and A0008 should both be displayed)
		this.analysisCenters = {
			A0003: { lon: 123.152480, lat: 32.345120 },
			A0008: { lon: 123.181261, lat: 32.327137 }
		};
        this.centerChangeRadiusMeters = 50.0;
        
        this.bindEvents();
        this.init();
    }

    /**
     * Initialize Cesium map
     */
    init() {
        // Configure Cesium access token (if needed)
        Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1YjFhYTRjZS0zYzZlLTRmN2ItOTE5NC1mMzEwYjFiZjE3NTUiLCJpZCI6MzEzNjA3LCJpYXQiOjE3NTAzMDY1MjF9.k7exedEe-OwSQ2qgC5NNIMec5tXhTiCEp6of6vdYv0o';

        // Create Cesium viewer with OSM basemap and default ellipsoid terrain
        this.viewer = new Cesium.Viewer('cesiumContainer', {
            imageryProvider: new Cesium.OpenStreetMapImageryProvider({
                url: 'https://a.tile.openstreetmap.org/'
            }),
            terrainProvider: new Cesium.EllipsoidTerrainProvider(),
            timeline: false,
            animation: false,
            sceneModePicker: false,
            baseLayerPicker: true, 
            geocoder: false,
            homeButton: false,
            infoBox: true,
            selectionIndicator: true,
            navigationHelpButton: false,
            navigationInstructionsInitiallyVisible: false
        });

        // Create data sources
        this.dataSource = new Cesium.CustomDataSource('vehicleTrack');
        this.viewer.dataSources.add(this.dataSource);
        
        this.topologyDataSource = new Cesium.CustomDataSource('intersectionTopology');
        this.viewer.dataSources.add(this.topologyDataSource);

        // Overlay data source for analysis markers (center + radius)
        this.overlayDataSource = new Cesium.CustomDataSource('analysisOverlays');
        this.viewer.dataSources.add(this.overlayDataSource);

        // Set initial view to China region
        this.viewer.camera.setView({
            destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 1000000)
        });

        // Load available dates for default road
        setTimeout(() => {
            this.loadAvailableDates();
            this.loadSpeedAvailableDates();
        }, 1000);

        // Ensure modal is hidden on initialization
        this.ensureModalHidden();

        // Draw analysis center and radius overlay
        this.drawAnalysisCenterOverlay();

        console.log('Cesium map initialization completed');
    }

    /**
     * Ensure modal is hidden on initialization
     */
    ensureModalHidden() {
        const timelineModal = document.getElementById('timelineModal');
        const speedModal = document.getElementById('speedModal');
        
        if (timelineModal) {
            timelineModal.style.display = 'none';
            timelineModal.style.visibility = 'hidden';
        }
        
        if (speedModal) {
            speedModal.style.display = 'none';
            speedModal.style.visibility = 'hidden';
        }
        
        document.body.style.overflow = 'auto';
    }

    /**
     * Helper to fetch JSON
     */
    async fetchJson(url) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    }

    /**
	 * Draw analysis center markers and radius circles for all configured roads
     */
    drawAnalysisCenterOverlay() {
		if (!this.overlayDataSource || !this.viewer) return;
		// Clear previous overlays
		this.overlayDataSource.entities.removeAll();

		const r = this.centerChangeRadiusMeters;

		Object.entries(this.analysisCenters).forEach(([roadId, center]) => {
			const lon = center.lon;
			const lat = center.lat;
			const centerPos = Cesium.Cartesian3.fromDegrees(lon, lat);

			// Choose distinct colors per road for clarity
			const pointColor = roadId === 'A0008' ? Cesium.Color.MAGENTA : Cesium.Color.CYAN;
			const outlineColor = roadId === 'A0008' ? Cesium.Color.YELLOW : Cesium.Color.RED;
			const fillColor = (roadId === 'A0008' ? Cesium.Color.MAGENTA : Cesium.Color.CYAN).withAlpha(0.15);

			// Center point marker with label
			this.overlayDataSource.entities.add({
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

			// Radius circle (meters)
			this.overlayDataSource.entities.add({
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

    /**
     * Draw a simple line chart for a generic metric over time using canvas
     */
    drawGenericMetricChart(containerEl, points, metricKey, title = '') {
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
                // Cast categorical/flag to number if possible; else 0
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

        // Axes
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Title
        ctx.fillStyle = '#000';
        ctx.font = '14px sans-serif';
        ctx.fillText(`${metricKey} over time`, padding, padding - 12);

        // Line
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
            const x = xMap(p.collectiontime);
            const y = yMap(values[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Dots
        ctx.fillStyle = '#007bff';
        points.forEach((p, i) => {
            const x = xMap(p.collectiontime);
            const y = yMap(values[i]);
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        const loadDataBtn = document.getElementById('loadDataBtn');
        const clearDataBtn = document.getElementById('clearDataBtn');
        const loadDatesBtn = document.getElementById('loadDatesBtn');
        const vehicleModeSelect = document.getElementById('vehicleMode');
        const roadIdSelect = document.getElementById('roadId');
        
        // Traffic light query elements
        const loadCyclesBtn = document.getElementById('loadCyclesBtn');
        const queryTrafficLightBtn = document.getElementById('queryTrafficLightBtn');
        const trafficRoadIdSelect = document.getElementById('trafficRoadId');
        const closeTimelineModal = document.getElementById('closeTimelineModal');
        const timelineModal = document.getElementById('timelineModal');
        
        // Speed analysis elements
        const loadSpeedDatesBtn = document.getElementById('loadSpeedDatesBtn');
        const analyzeSpeedBtn = document.getElementById('analyzeSpeedBtn');
        const speedRoadIdSelect = document.getElementById('speedRoadId');
        const speedModeSelect = document.getElementById('speedMode');
        const loadTimeRangeBtn = document.getElementById('loadTimeRangeBtn');
        const closeSpeedModal = document.getElementById('closeSpeedModal');
        const speedModal = document.getElementById('speedModal');
        const speedMetricSelect = document.getElementById('speedMetric');
        const speedSegIdInput = document.getElementById('speedSegId');
        
        // Topology elements
        const loadTopologyBtn = document.getElementById('loadTopologyBtn');
        const clearTopologyBtn = document.getElementById('clearTopologyBtn');
        const topologyRoadIdSelect = document.getElementById('topologyRoadId');


        loadDataBtn.addEventListener('click', () => {
            this.loadVehicleData();
        });

        clearDataBtn.addEventListener('click', () => {
            this.clearData();
        });

        loadDatesBtn.addEventListener('click', () => {
            this.loadAvailableDates();
        });

        // Handle vehicle mode switching
        vehicleModeSelect.addEventListener('change', (e) => {
            this.toggleVehicleMode(e.target.value);
        });

        // Vehicle time range button
        const loadVehicleTimeRangeBtn = document.getElementById('loadVehicleTimeRangeBtn');
        if (loadVehicleTimeRangeBtn) {
            loadVehicleTimeRangeBtn.addEventListener('click', () => {
                this.loadVehicleTimeRange();
            });
        }

        // Handle road ID change to auto-load dates
        roadIdSelect.addEventListener('change', () => {
            this.loadAvailableDates();
        });

        // Traffic light query event listeners
        loadCyclesBtn.addEventListener('click', () => {
            this.loadAvailableCycles();
        });

        queryTrafficLightBtn.addEventListener('click', () => {
            this.queryTrafficLightStatus();
        });

        // Speed analysis event listeners
        loadSpeedDatesBtn.addEventListener('click', () => {
            this.loadSpeedAvailableDates();
        });

        analyzeSpeedBtn.addEventListener('click', () => {
            this.analyzeSpeed();
        });

        // Handle speed road ID change to auto-load dates
        speedRoadIdSelect.addEventListener('change', () => {
            this.loadSpeedAvailableDates();
        });

        // Handle speed mode switching
        speedModeSelect.addEventListener('change', (e) => {
            this.toggleSpeedMode(e.target.value);
        });

        // Load time range button
        loadTimeRangeBtn.addEventListener('click', () => {
            this.loadTimeRange();
        });

        // Track metric and seg id selections
        if (speedMetricSelect) {
            this.currentSelectedMetric = speedMetricSelect.value || 'speed';
            speedMetricSelect.addEventListener('change', (e) => {
                this.currentSelectedMetric = e.target.value || 'speed';
            });
        }
        if (speedSegIdInput) {
            speedSegIdInput.addEventListener('input', (e) => {
                const v = e.target.value;
                this.currentSelectedSegId = v ? Number(v) : null;
            });
        }

        // Topology event listeners
        loadTopologyBtn.addEventListener('click', () => {
            this.loadIntersectionTopology();
        });

        clearTopologyBtn.addEventListener('click', () => {
            this.clearTopology();
        });

        // Timeline modal event listeners
        if (closeTimelineModal) {
            closeTimelineModal.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.closeTimelineModal();
            });
        }

        // Close modal when clicking outside
        if (timelineModal) {
            timelineModal.addEventListener('click', (e) => {
                if (e.target === timelineModal) {
                    this.closeTimelineModal();
                }
            });
        }

        // Speed modal event listeners
        if (closeSpeedModal) {
            closeSpeedModal.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.closeSpeedModal();
            });
        }

        // Close speed modal when clicking outside
        if (speedModal) {
            speedModal.addEventListener('click', (e) => {
                if (e.target === speedModal) {
                    this.closeSpeedModal();
                }
            });
        }

        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const timelineModal = document.getElementById('timelineModal');
                const speedModal = document.getElementById('speedModal');
                
                if (timelineModal && timelineModal.style.display === 'block') {
                    this.closeTimelineModal();
                } else if (speedModal && speedModal.style.display === 'block') {
                    this.closeSpeedModal();
                }
            }
        });

        // Panel collapse functionality
        document.querySelectorAll('.panel-header').forEach(header => {
            header.addEventListener('click', () => {
                this.togglePanel(header.dataset.panel);
            });
        });

    }

    /**
     * Toggle panel collapse/expand
     */
    togglePanel(panelId) {
        const panel = document.getElementById(panelId);
        const indicator = panel.querySelector('.collapse-indicator');
        
        if (panel.classList.contains('collapsed')) {
            panel.classList.remove('collapsed');
            indicator.textContent = '−';
        } else {
            panel.classList.add('collapsed');
            indicator.textContent = '+';
        }
    }



    /**
     * Toggle between single, batch, and time range vehicle selection modes
     */
    toggleVehicleMode(mode) {
        const singleSection = document.getElementById('singleVehicleSection');
        const batchSection = document.getElementById('batchVehicleSection');
        const timeRangeSection = document.getElementById('timeRangeVehicleSection');

        if (mode === 'single') {
            singleSection.style.display = 'block';
            batchSection.style.display = 'none';
            timeRangeSection.style.display = 'none';
        } else if (mode === 'batch') {
            singleSection.style.display = 'none';
            batchSection.style.display = 'block';
            timeRangeSection.style.display = 'none';
        } else if (mode === 'time_range') {
            singleSection.style.display = 'none';
            batchSection.style.display = 'none';
            timeRangeSection.style.display = 'block';
        }
    }

    /**
     * Toggle between single and time range speed analysis modes
     */
    toggleSpeedMode(mode) {
        const singleSection = document.getElementById('singleSpeedSection');
        const timeRangeSection = document.getElementById('timeRangeSpeedSection');

        if (mode === 'single') {
            singleSection.style.display = 'block';
            timeRangeSection.style.display = 'none';
        } else {
            singleSection.style.display = 'none';
            timeRangeSection.style.display = 'block';
        }
    }

    /**
     * Show status information
     */
    showStatus(message, isLoading = false) {
        const statusText = document.getElementById('statusText');
        const loadingSpinner = document.getElementById('loadingSpinner');

        statusText.textContent = message;
        loadingSpinner.style.display = isLoading ? 'block' : 'none';
    }

    /**
     * Update data information display
     */
    updateDataInfo(summary) {
        if (summary && summary.status === 'success') {
            document.getElementById('totalPoints').textContent = `Total Points: ${summary.total_points}`;
            document.getElementById('timeRange').textContent = 
                `Time Range: ${summary.time_range.start} - ${summary.time_range.end}`;
            document.getElementById('speedRange').textContent = 
                `Speed Range: ${summary.speed_stats.min.toFixed(1)} - ${summary.speed_stats.max.toFixed(1)} km/h`;
        }
    }

    /**
     * Load time range information for the selected date (Vehicle Panel)
     */
    async loadVehicleTimeRange() {
        try {
            const roadId = document.getElementById('roadId').value;
            const date = document.getElementById('dateFilter').value;
            const timeRangeInfo = document.getElementById('vehicleTimeRangeInfo');
            
            if (!date) {
                this.showStatus('Please select a date first');
                return;
            }
            
            timeRangeInfo.textContent = 'Loading time range...';
            
            const response = await fetch(`${this.apiBaseUrl}/speed/time-range?road_id=${roadId}&date=${date}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                const startTimeInput = document.getElementById('vehicleStartTime');
                const endTimeInput = document.getElementById('vehicleEndTime');
                
                // Convert HH:MM:SS to HH:MM for time inputs
                const minTime = result.min_time.substring(0, 5);
                const maxTime = result.max_time.substring(0, 5);
                
                // Set default values to the available range
                startTimeInput.value = minTime;
                endTimeInput.value = maxTime;
                
                // Set min/max attributes for validation
                startTimeInput.min = minTime;
                startTimeInput.max = maxTime;
                endTimeInput.min = minTime;
                endTimeInput.max = maxTime;
                
                timeRangeInfo.innerHTML = `
                    <strong>Available Time Range:</strong> ${minTime} - ${maxTime} (${result.time_range_duration})<br>
                    <strong>Total Vehicles:</strong> ${result.total_vehicles} | <strong>Total Records:</strong> ${result.total_records}
                `;
                
                this.showStatus(`Vehicle time range loaded for ${date}`);
            } else {
                throw new Error(result.message || 'Failed to load time range');
            }
            
        } catch (error) {
            console.error('Error loading vehicle time range:', error);
            const timeRangeInfo = document.getElementById('vehicleTimeRangeInfo');
            timeRangeInfo.textContent = `Error: ${error.message}`;
            this.showStatus(`Failed to load time range: ${error.message}`);
        }
    }

    /**
     * Load available dates for the selected road
     */
    async loadAvailableDates() {
        try {
            const roadId = document.getElementById('roadId').value;
            const dateSelect = document.getElementById('dateFilter');
            
            this.showStatus('Loading available dates...', true);
            
            const response = await fetch(`${this.apiBaseUrl}/vehicle/dates?road_id=${roadId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Clear existing options
                dateSelect.innerHTML = '<option value="" selected>All Dates</option>';
                
                // Add new date options
                result.dates.forEach(date => {
                    const option = document.createElement('option');
                    option.value = date;
                    option.textContent = date;
                    dateSelect.appendChild(option);
                });
                
                this.showStatus(`Loaded ${result.total_dates} available dates`);
            } else {
                throw new Error(result.message || 'Failed to load dates');
            }
            
        } catch (error) {
            console.error('Error loading dates:', error);
            this.showStatus(`Failed to load dates: ${error.message}`);
        }
    }

    /**
     * Load vehicle data from backend
     */
    async loadVehicleData() {
        try {
            this.showStatus('Loading data...', true);

            const vehicleMode = document.getElementById('vehicleMode').value;
            const roadId = document.getElementById('roadId').value;
            const dataLimit = document.getElementById('dataLimit').value;
            const dateFilter = document.getElementById('dateFilter').value;

            let apiUrl;
            let summaryUrl;

            if (vehicleMode === 'single') {
                // Single vehicle mode
                const vehicleId = document.getElementById('vehicleId').value;
                apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_id=${vehicleId}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?vehicle_id=${vehicleId}&road_id=${roadId}`;
            } else if (vehicleMode === 'batch') {
                // Batch vehicle mode
                const vehicleCount = document.getElementById('vehicleCount').value;
                const direction = document.getElementById('batchDirection').value;
                apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_count=${vehicleCount}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?vehicle_count=${vehicleCount}&road_id=${roadId}`;
                
                // Add direction filter if specified
                if (direction) {
                    apiUrl += `&direction=${direction}`;
                    summaryUrl += `&direction=${direction}`;
                }
            } else if (vehicleMode === 'time_range') {
                // Time range mode
                const startTime = document.getElementById('vehicleStartTime').value;
                const endTime = document.getElementById('vehicleEndTime').value;
                
                if (!dateFilter) {
                    this.showStatus('Please select a date first');
                    return;
                }
                
                if (!startTime || !endTime) {
                    this.showStatus('Please select start and end time');
                    return;
                }
                
                if (startTime >= endTime) {
                    this.showStatus('End time must be after start time');
                    return;
                }
                
                apiUrl = `${this.apiBaseUrl}/vehicle/data?start_time=${startTime}&end_time=${endTime}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?start_time=${startTime}&end_time=${endTime}&road_id=${roadId}`;
                
                // Add direction filter if specified
                const direction = document.getElementById('timeRangeDirection').value;
                if (direction) {
                    apiUrl += `&direction=${direction}`;
                    summaryUrl += `&direction=${direction}`;
                }
            }

            // Add date filter if specified
            if (dateFilter) {
                apiUrl += `&date=${dateFilter}`;
                summaryUrl += `&date=${dateFilter}`;
            }

            if (dataLimit && vehicleMode !== 'time_range') {
                apiUrl += `&limit=${dataLimit}`;
            }

            // Send request to get data
            const response = await fetch(apiUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }

            const result = await response.json();

            if (result.status === 'success') {
                this.currentVehicleData = result.data;
                
                // Get data summary
                const summaryResponse = await fetch(summaryUrl);
                const summaryResult = await summaryResponse.json();
                
                // Visualize data
                this.visualizeData(this.currentVehicleData);
                this.updateDataInfo(summaryResult);
                
                const totalPoints = Array.isArray(result.data) ? result.data.length : result.total_points;
                const dateFilterText = dateFilter ? ` (date: ${dateFilter})` : '';
                
                let modeText;
                if (vehicleMode === 'single') {
                    modeText = '1 vehicle';
                } else if (vehicleMode === 'batch') {
                    modeText = `${result.vehicle_count} vehicles`;
                } else if (vehicleMode === 'time_range') {
                    const startTime = document.getElementById('vehicleStartTime').value;
                    const endTime = document.getElementById('vehicleEndTime').value;
                    modeText = `${result.vehicle_count || 'multiple'} vehicles (${startTime}-${endTime})`;
                }
                
                this.showStatus(`Successfully loaded ${totalPoints} data points from ${modeText}${dateFilterText}`);
            } else {
                throw new Error(result.message || 'Failed to load data');
            }

        } catch (error) {
            console.error('Error loading data:', error);
            this.showStatus(`Loading failed: ${error.message}`);
        }
    }

    /**
     * Visualize data on the map
     */
    visualizeData(data) {
        if (!data || data.length === 0) {
            console.warn('No data to visualize');
            return;
        }

        // Clear previous data
        this.dataSource.entities.removeAll();

        // Group data by vehicle ID + date + seg_id (if present)
        const vehicleGroups = {};
        data.forEach(point => {
            // Create unique key combining vehicle_id, date, and seg_id
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
        const colors = this.getVehicleColors(trajectoryKeys.length);

        // Process each trajectory separately (vehicle+date combination)
        trajectoryKeys.forEach((trajectoryKey, trajectoryIndex) => {
            const trajectoryInfo = vehicleGroups[trajectoryKey];
            const trajectoryData = trajectoryInfo.points;
            const trajectoryColor = colors[trajectoryIndex];
            const positions = [];

            // Create entity for each data point of this trajectory
            trajectoryData.forEach((point, index) => {
                const position = Cesium.Cartesian3.fromDegrees(
                    point.longitude, 
                    point.latitude, 
                    0
                );

                positions.push(position);

                // Determine point color: use trajectory color modulated by speed
                let pointColor = this.getSpeedColor(point.speed);
                if (trajectoryKeys.length > 1) {
                    // In multi-trajectory mode, blend speed color with trajectory color
                    pointColor = this.blendColors(trajectoryColor, pointColor);
                }

                // Create point entity
                const pointEntity = this.dataSource.entities.add({
                    position: position,
                    point: {
                        pixelSize: 8,
                        color: pointColor,
                        outlineColor: Cesium.Color.WHITE,
                        outlineWidth: 2,
                        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
                    },
                    description: this.createPointDescription(point, index, trajectoryKeys.length > 1, trajectoryInfo.date)
                });
            });

            // Create trajectory line for this trajectory
            if (positions.length > 1) {
                this.dataSource.entities.add({
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

        // Fly to data range
        this.viewer.flyTo(this.dataSource);

        console.log(`Visualized ${data.length} data points from ${trajectoryKeys.length} trajectory(ies)`);
    }

    /**
     * Get color based on speed
     */
    getSpeedColor(speed) {
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

    /**
     * Create point description information
     */
    createPointDescription(point, index, isMultiTrajectory = false, pointDate = null) {
        const title = isMultiTrajectory ? 
            `Vehicle ${point.vehicle_id} (${pointDate || point.date}) - Point #${index + 1}` : 
            `Data Point #${index + 1}`;
            
        return `
            <div class="point-info">
                <h4>${title}</h4>
                <table>
                    <tr><td>Vehicle ID:</td><td>${point.vehicle_id}</td></tr>
                    ${typeof point.seg_id === 'number' ? `<tr><td>Seg ID:</td><td>${point.seg_id}</td></tr>` : ''}
                    <tr><td>Date:</td><td>${point.date}</td></tr>
                    <tr><td>Time:</td><td>${point.time_stamp}</td></tr>
                    <tr><td>Road ID:</td><td>${point.road_id}</td></tr>
                    <tr><td>Longitude:</td><td>${point.longitude.toFixed(6)}</td></tr>
                    <tr><td>Latitude:</td><td>${point.latitude.toFixed(6)}</td></tr>
                    <tr><td>Speed:</td><td>${point.speed.toFixed(1)} km/h</td></tr>
                    <tr><td>Accelerator:</td><td>${point.acceleratorpedal}%</td></tr>
                    <tr><td>Brake Status:</td><td>${point.brakestatus}</td></tr>
                    <tr><td>Gear Number:</td><td>${point.gearnum || 'N/A'}</td></tr>
                    <tr><td>Have Brake:</td><td>${point.havebrake || 'N/A'}</td></tr>
                    <tr><td>Have Driver:</td><td>${point.havedriver || 'N/A'}</td></tr>
                </table>
            </div>
        `;
    }

    /**
     * Get distinct colors for different vehicles
     */
    getVehicleColors(vehicleCount) {
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
                // Generate more colors by cycling through hues
                const hue = (i * 137.5) % 360; // Golden angle for good distribution
                colors.push(Cesium.Color.fromHsl(hue / 360, 0.7, 0.6));
            }
        }
        return colors;
    }

    /**
     * Blend two colors together
     */
    blendColors(vehicleColor, speedColor, ratio = 0.7) {
        // Blend vehicle color (70%) with speed color (30%)
        return new Cesium.Color(
            vehicleColor.red * ratio + speedColor.red * (1 - ratio),
            vehicleColor.green * ratio + speedColor.green * (1 - ratio),
            vehicleColor.blue * ratio + speedColor.blue * (1 - ratio),
            1.0
        );
    }

    /**
     * Load available cycles for traffic lights
     */
    async loadAvailableCycles() {
        try {
            const roadId = document.getElementById('trafficRoadId').value;
            
            this.showTrafficStatus('Loading available cycles...', 'info');
            
            const response = await fetch(`${this.apiBaseUrl}/traffic/cycles?road_id=${roadId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateCycleInfo(data.cycles);
                this.showTrafficStatus(`Loaded ${data.total_cycles} cycles for ${roadId}`, 'success');
            } else {
                this.showTrafficStatus(`Error: ${data.message}`, 'error');
            }
            
        } catch (error) {
            console.error('Error loading available cycles:', error);
            this.showTrafficStatus('Failed to load available cycles', 'error');
        }
    }
    
    /**
     * Update cycle information display
     */
    updateCycleInfo(cycles) {
        const cycleInput = document.getElementById('cycleNum');
        if (cycles && cycles.length > 0) {
            cycleInput.min = Math.min(...cycles);
            cycleInput.max = Math.max(...cycles);
            cycleInput.value = cycles[0];
        }
    }
    
    /**
     * Query traffic light status for specific cycle
     */
    async queryTrafficLightStatus() {
        try {
            const roadId = document.getElementById('trafficRoadId').value;
            const cycleNum = document.getElementById('cycleNum').value;
            
            if (!cycleNum) {
                this.showTrafficStatus('Please enter a cycle number', 'warning');
                return;
            }
            
            this.showTrafficStatus('Querying traffic light status...', 'info');
            
            const response = await fetch(`${this.apiBaseUrl}/traffic/status?road_id=${roadId}&cycle_num=${cycleNum}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.currentTrafficLightData = data; // Store the data
                this.updateTrafficLightStatus(data);
                this.prepareTrafficLightResults(data);
                this.showTrafficStatus(`Loaded traffic light data for ${roadId} cycle ${cycleNum}`, 'success');
            } else {
                this.showTrafficStatus(`Error: ${data.message}`, 'error');
            }
            
        } catch (error) {
            console.error('Error querying traffic light status:', error);
            this.showTrafficStatus('Failed to query traffic light status', 'error');
        }
    }
    
    /**
     * Update traffic light status in the panel
     */
    updateTrafficLightStatus(data) {
        const statusDiv = document.getElementById('trafficLightStatus');
        
        statusDiv.innerHTML = `
            <div class="status-message success">
                <strong>✓ Query Successful</strong><br>
                Road: ${data.road_id} | Cycle: ${data.cycle_num}<br>
                ${data.total_phases} phases found<br>
                <button id="viewTimelineBtn" class="btn btn-primary" style="margin-top: 10px;">
                    View Timeline Chart
                </button>
            </div>
        `;
        
        // Add event listener for view timeline button
        document.getElementById('viewTimelineBtn').addEventListener('click', () => {
            this.openTimelineModal();
        });
    }
    
    /**
     * Open timeline modal
     */
    openTimelineModal() {
        const modal = document.getElementById('timelineModal');
        
        if (this.currentTrafficLightData && modal) {
            this.displayTrafficLightResults(this.currentTrafficLightData);
            modal.style.display = 'block';
            modal.style.visibility = 'visible';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        } else {
            this.showTrafficStatus('No traffic light data to display', 'warning');
        }
    }
    
    /**
     * Close timeline modal
     */
    closeTimelineModal() {
        const modal = document.getElementById('timelineModal');
        if (modal) {
            modal.style.display = 'none';
            modal.style.visibility = 'hidden';
            document.body.style.overflow = 'auto'; // Restore scrolling
        }
    }

    /**
     * Prepare traffic light data (called after successful query)
     */
    prepareTrafficLightResults(data) {
        // Data is stored in this.currentTrafficLightData, ready for display in modal
        console.log('Traffic light data prepared for display:', data);
    }

    /**
     * Display traffic light query results with timeline chart in modal
     */
    displayTrafficLightResults(data) {
        const titleElement = document.getElementById('timelineModalTitle');
        const dataDiv = document.getElementById('timelineModalData');
        
        // Update modal title
        titleElement.textContent = `Traffic Light Timeline - ${data.road_id} Cycle ${data.cycle_num}`;
        
        // Sort phases by phase_id for consistent display
        const sortedPhases = data.phases.sort((a, b) => a.phase_id.localeCompare(b.phase_id));
        
        // Calculate time range for the entire cycle
        const startTimes = sortedPhases.map(p => new Date(p.start_time));
        const endTimes = sortedPhases.map(p => new Date(p.end_time));
        const cycleStart = new Date(Math.min(...startTimes));
        const cycleEnd = new Date(Math.max(...endTimes));
        const cycleDuration = cycleEnd - cycleStart;
        
        // Create timeline visualization
        let html = `
            <div class="cycle-summary">
                <h5>Cycle ${data.cycle_num} Summary (${data.road_id})</h5>
                <p><strong>Cycle Duration:</strong> ${Math.round(cycleDuration / 1000)} seconds</p>
                <p><strong>Time Range:</strong> ${this.formatTime(cycleStart)} - ${this.formatTime(cycleEnd)}</p>
            </div>
            <div class="timeline-container">
                <div class="timeline-header">
                    <div class="timeline-title">Traffic Light Timeline (Red Light Periods)</div>
                </div>
                <div class="timeline-chart">
        `;
        
        // Calculate timeline scale
        const timelineWidth = 100; // percentage
        
        sortedPhases.forEach((phase, index) => {
            const phaseStart = new Date(phase.start_time);
            const phaseEnd = new Date(phase.end_time);
            const duration = Math.round((phaseEnd - phaseStart) / 1000);
            
            // Calculate position and width as percentage of total cycle
            const startOffset = ((phaseStart - cycleStart) / cycleDuration) * timelineWidth;
            const width = ((phaseEnd - phaseStart) / cycleDuration) * timelineWidth;
            
            html += `
                <div class="timeline-row">
                    <div class="phase-label">Phase ${phase.phase_id}</div>
                    <div class="timeline-bar-container">
                        <div class="timeline-bar" style="left: ${startOffset}%; width: ${width}%; background-color: #e74c3c;">
                            <div class="timeline-bar-info">
                                <span class="duration-text">${duration}s</span>
                            </div>
                        </div>
                        <div class="time-markers">
                            <span class="start-time" style="left: ${startOffset}%;">${this.formatTime(phaseStart)}</span>
                            <span class="end-time" style="left: ${startOffset + width}%;">${this.formatTime(phaseEnd)}</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
            <div class="phase-details-list">
                <h5>Phase Details</h5>
        `;
        
        // Add detailed phase information
        sortedPhases.forEach((phase, index) => {
            const phaseStart = new Date(phase.start_time);
            const phaseEnd = new Date(phase.end_time);
            const duration = Math.round((phaseEnd - phaseStart) / 1000);
            
            html += `
                <div class="phase-item">
                    <div class="phase-header">
                        <div class="phase-indicator"></div>
                        Phase ${phase.phase_id}
                    </div>
                    <div class="phase-details">
                        <div class="time-info">Start: ${this.formatTime(phaseStart)}</div>
                        <div class="time-info">End: ${this.formatTime(phaseEnd)}</div>
                        <div class="duration-info">Duration: ${duration} seconds</div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        dataDiv.innerHTML = html;
    }
    
    /**
     * Format time for display
     */
    formatTime(date) {
        return date.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
    
    /**
     * Show status message in traffic light panel
     */
    showTrafficStatus(message, type = 'info') {
        const statusDiv = document.getElementById('trafficStatus');
        const statusText = document.getElementById('trafficStatusText');
        
        statusText.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        // Auto-hide after 3 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }

    /**
     * Clear data on the map
     */
    clearData() {
        if (this.dataSource) {
            this.dataSource.entities.removeAll();
        }
        
        this.currentVehicleData = [];
        
        // Reset data information display
        document.getElementById('totalPoints').textContent = 'Total Points: --';
        document.getElementById('timeRange').textContent = 'Time Range: --';
        document.getElementById('speedRange').textContent = 'Speed Range: --';
        
        this.showStatus('Data cleared');
        
        console.log('All data cleared');
    }

    /**
     * Load available dates for speed analysis
     */
    async loadSpeedAvailableDates() {
        try {
            const roadId = document.getElementById('speedRoadId').value;
            const dateSelect = document.getElementById('speedDate');
            
            this.showSpeedStatus('Loading available dates...', 'info');
            
            const response = await fetch(`${this.apiBaseUrl}/vehicle/dates?road_id=${roadId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Clear existing options
                dateSelect.innerHTML = '<option value="" selected>Select Date</option>';
                
                // Add new date options
                result.dates.forEach(date => {
                    const option = document.createElement('option');
                    option.value = date;
                    option.textContent = date;
                    dateSelect.appendChild(option);
                });
                
                this.showSpeedStatus(`Loaded ${result.total_dates} available dates`, 'success');
            } else {
                throw new Error(result.message || 'Failed to load dates');
            }
            
        } catch (error) {
            console.error('Error loading speed analysis dates:', error);
            this.showSpeedStatus(`Failed to load dates: ${error.message}`, 'error');
        }
    }

    

    /**
     * Load time range information for the selected date
     */
    async loadTimeRange() {
        try {
            const roadId = document.getElementById('speedRoadId').value;
            const date = document.getElementById('speedDate').value;
            const timeRangeInfo = document.getElementById('timeRangeInfo');
            
            if (!date) {
                this.showSpeedStatus('Please select a date first', 'warning');
                return;
            }
            
            timeRangeInfo.textContent = 'Loading time range...';
            
            const response = await fetch(`${this.apiBaseUrl}/speed/time-range?road_id=${roadId}&date=${date}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                const startTimeInput = document.getElementById('speedStartTime');
                const endTimeInput = document.getElementById('speedEndTime');
                
                // Convert HH:MM:SS to HH:MM for time inputs
                const minTime = result.min_time.substring(0, 5);
                const maxTime = result.max_time.substring(0, 5);
                
                // Set default values to the available range
                startTimeInput.value = minTime;
                endTimeInput.value = maxTime;
                
                // Set min/max attributes for validation
                startTimeInput.min = minTime;
                startTimeInput.max = maxTime;
                endTimeInput.min = minTime;
                endTimeInput.max = maxTime;
                
                timeRangeInfo.innerHTML = `
                    <strong>Available Time Range:</strong> ${minTime} - ${maxTime} (${result.time_range_duration})<br>
                    <strong>Total Vehicles:</strong> ${result.total_vehicles} | <strong>Total Records:</strong> ${result.total_records}
                `;
                
                this.showSpeedStatus(`Time range loaded for ${date}`, 'success');
            } else {
                throw new Error(result.message || 'Failed to load time range');
            }
            
        } catch (error) {
            console.error('Error loading time range:', error);
            const timeRangeInfo = document.getElementById('timeRangeInfo');
            timeRangeInfo.textContent = `Error: ${error.message}`;
            this.showSpeedStatus(`Failed to load time range: ${error.message}`, 'error');
        }
    }

    /**
     * Analyze speed for specific vehicle or time range
     */
    async analyzeSpeed() {
        try {
            const roadId = document.getElementById('speedRoadId').value;
            const speedMode = document.getElementById('speedMode').value;
            const date = document.getElementById('speedDate').value;
            const selectedMetric = this.currentSelectedMetric || 'speed';
            const segId = this.currentSelectedSegId;
            
            if (!date) {
                this.showSpeedStatus('Please select a date', 'warning');
                return;
            }
            
            let apiUrl;
            
            if (speedMode === 'single') {
                // Single vehicle mode
                const vehicleId = document.getElementById('speedVehicleId').value;
                const direction = document.getElementById('singleSpeedDirection').value;
                
                if (!vehicleId) {
                    this.showSpeedStatus('Please enter a vehicle ID', 'warning');
                    return;
                }
                
                apiUrl = `${this.apiBaseUrl}/speed/analysis?road_id=${roadId}&vehicle_id=${vehicleId}&date=${date}`;
                
                // Add direction filter if specified
                if (direction) {
                    apiUrl += `&direction=${direction}`;
                }
            } else {
                // Time range mode
                const startTime = document.getElementById('speedStartTime').value;
                const endTime = document.getElementById('speedEndTime').value;
                const direction = document.getElementById('timeRangeSpeedDirection').value;
                
                if (!startTime || !endTime) {
                    this.showSpeedStatus('Please select start and end time', 'warning');
                    return;
                }
                
                if (startTime >= endTime) {
                    this.showSpeedStatus('End time must be after start time', 'warning');
                    return;
                }
                
                apiUrl = `${this.apiBaseUrl}/speed/analysis?road_id=${roadId}&start_time=${startTime}&end_time=${endTime}&date=${date}`;
                
                // Add direction filter if specified
                if (direction) {
                    apiUrl += `&direction=${direction}`;
                }
            }
            // Optional seg_id filter
            if (typeof segId === 'number' && Number.isFinite(segId)) {
                apiUrl += `&seg_id=${segId}`;
            }
            
            this.showSpeedStatus('Analyzing speed data...', 'info');
            
            // Get speed analysis data
            const speedResponse = await fetch(apiUrl);
            const speedData = await speedResponse.json();
            
            console.log('Speed analysis response:', speedData);
            
            if (speedData.status !== 'success') {
                throw new Error(speedData.message || 'Failed to get speed data');
            }
            
            if (!speedData.data || speedData.data.length === 0) {
                const errorMsg = speedMode === 'single' ? 
                    'No speed data found for the specified vehicle and date' :
                    'No speed data found for the specified time range and date';
                throw new Error(errorMsg);
            }
            
            // Optional client-side filter by seg_id if available in records
            if (typeof segId === 'number' && Number.isFinite(segId)) {
                const beforeCount = speedData.data.length;
                speedData.data = speedData.data.filter(p => p && p.seg_id === segId);
                console.log(`Filtered by seg_id=${segId}: ${beforeCount} -> ${speedData.data.length}`);
                if (speedData.data.length === 0) {
                    throw new Error('No data after applying seg_id filter');
                }
            }

            // Attach selected metric
            speedData.metric = selectedMetric;
            this.currentSpeedData = speedData;
            
            // Get traffic light data for the time range
            if (speedData.data && speedData.data.length > 0) {
                const startTime = speedData.data[0].time_stamp;
                const endTime = speedData.data[speedData.data.length - 1].time_stamp;
                
                const trafficResponse = await fetch(`${this.apiBaseUrl}/speed/traffic-lights?road_id=${roadId}&start_time=${startTime}&end_time=${endTime}`);
                const trafficData = await trafficResponse.json();
                
                console.log('Traffic lights response:', trafficData);
                
                if (trafficData.status === 'success') {
                    this.currentSpeedTrafficLights = trafficData;
                } else {
                    console.warn('Failed to get traffic light data:', trafficData.message);
                    this.currentSpeedTrafficLights = null;
                }
            }
            
            // Update status display
            this.updateSpeedAnalysisStatus(speedData);
            this.prepareSpeedResults(speedData);
            
            let successMsg;
            if (speedData.mode === 'single') {
                successMsg = `Speed analysis completed for vehicle ${speedData.vehicle_id}`;
            } else {
                successMsg = `Speed analysis completed for ${speedData.vehicle_count} vehicles in time range ${speedData.start_time}-${speedData.end_time}`;
            }
            this.showSpeedStatus(successMsg, 'success');
            
        } catch (error) {
            console.error('Error analyzing speed:', error);
            this.showSpeedStatus(`Speed analysis failed: ${error.message}`, 'error');
        }
    }

    /**
     * Update speed analysis status in the panel
     */
    updateSpeedAnalysisStatus(data) {
        const statusDiv = document.getElementById('speedAnalysisStatus');
        
        let analysisInfo;
        if (data.mode === 'single') {
            analysisInfo = `Vehicle: ${data.vehicle_id}`;
        } else {
            // Time range mode
            const vehicleList = data.vehicle_ids.length > 5 ? 
                `${data.vehicle_ids.slice(0, 5).join(', ')}...` : 
                data.vehicle_ids.join(', ');
            analysisInfo = `Time Range: ${data.start_time} - ${data.end_time}<br>Vehicles (${data.vehicle_count}): ${vehicleList}`;
        }
        
        statusDiv.innerHTML = `
            <div class="status-message success">
                <strong>✓ Analysis Complete</strong><br>
                ${analysisInfo}<br>
                Road: ${data.road_id} | Date: ${data.date}<br>
                ${data.total_points} speed records found<br>
                <button id="viewSpeedChartBtn" class="btn btn-primary" style="margin-top: 10px;">
                    View Speed Chart
                </button>
            </div>
        `;
        
        // Add event listener for view chart button
        document.getElementById('viewSpeedChartBtn').addEventListener('click', () => {
            this.openSpeedModal();
        });
    }

    /**
     * Open speed analysis modal
     */
    openSpeedModal() {
        const modal = document.getElementById('speedModal');
        
        if (this.currentSpeedData && modal) {
            modal.style.display = 'block';
            modal.style.visibility = 'visible';
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
            
            // Wait for modal to be fully displayed before drawing chart
            setTimeout(() => {
                this.displaySpeedChart(this.currentSpeedData, this.currentSpeedTrafficLights);
            }, 200);
        } else {
            this.showSpeedStatus('No speed data to display', 'warning');
        }
    }

    /**
     * Close speed analysis modal
     */
    closeSpeedModal() {
        const modal = document.getElementById('speedModal');
    if (modal) {
        modal.style.display = 'none';
        modal.style.visibility = 'hidden';
            document.body.style.overflow = 'auto'; // Restore scrolling
        }
    }

    /**
     * Prepare speed analysis data (called after successful analysis)
     */
    prepareSpeedResults(data) {
        // Data is stored in this.currentSpeedData, ready for display in modal
        console.log('Speed analysis data prepared for display:', data);
    }

    /**
     * Display speed chart in modal
     */
    displaySpeedChart(speedData, trafficLightData) {
        const titleElement = document.getElementById('speedModalTitle');
        const chartContainer = document.getElementById('speedChartContainer');
        
        // Update modal title based on mode
        let titleText, infoText;
        if (speedData.mode === 'single') {
            titleText = `Metric Analysis (${speedData.metric}) - Vehicle ${speedData.vehicle_id} (${speedData.date})`;
            infoText = `Vehicle ${speedData.vehicle_id} on ${speedData.date} | Road ${speedData.road_id} | ${speedData.total_points} records`;
        } else {
            // Time range mode
            titleText = `Metric Analysis (${speedData.metric}) - Time Range ${speedData.start_time}-${speedData.end_time} (${speedData.date})`;
            infoText = `${speedData.vehicle_count} vehicles from ${speedData.start_time} to ${speedData.end_time} on ${speedData.date} | Road ${speedData.road_id} | ${speedData.total_points} records`;
        }
        
        titleElement.textContent = titleText;
        
        // Create chart header
        let html = `
            <div class="speed-chart-header">
                <div class="speed-chart-title">${speedData.metric} vs Time</div>
                <div class="speed-chart-info">
                    ${infoText}
                </div>
            </div>
            <div class="speed-chart-canvas" id="speedChartCanvas"></div>
        `;
        
        // Add legend
        html += `<div class="speed-analysis-legend">`;
        
        if (speedData.mode === 'single') {
            html += `
                <div class="speed-legend-item">
                    <div class="speed-legend-color" style="background-color: #4CAF50;"></div>
                    <span>${speedData.metric}</span>
                </div>`;
        } else {
            // Time range mode - show legend for multiple vehicles
            const vehicleColors = this.getVehicleColors(speedData.vehicle_count);
            speedData.vehicle_ids.forEach((vehicleId, index) => {
                const color = vehicleColors[index];
                const colorStr = `rgb(${Math.floor(color.red * 255)}, ${Math.floor(color.green * 255)}, ${Math.floor(color.blue * 255)})`;
                html += `
                    <div class="speed-legend-item">
                        <div class="speed-legend-color" style="background-color: ${colorStr};"></div>
                        <span>Vehicle ${vehicleId}</span>
                    </div>`;
            });
        }
        
        if (trafficLightData && trafficLightData.traffic_lights.length > 0) {
            html += `
                <div class="speed-legend-item">
                    <div class="speed-legend-marker" style="background-color: rgba(255, 0, 0, 0.3); border: 1px solid #dc3545;"></div>
                    <span>Red Light Periods</span>
                </div>`;
        }
        
        html += `</div>`;
        
        chartContainer.innerHTML = html;
        
        // Delay chart drawing to ensure container has correct dimensions
        setTimeout(() => {
            this.drawSpeedChart(speedData.data, trafficLightData, speedData.mode, speedData.vehicle_ids || []);
        }, 100);
    }

    /**
     * Draw speed chart using canvas
     */
    drawSpeedChart(speedPoints, trafficLightData, mode = 'single', vehicleIds = []) {
        const canvasContainer = document.getElementById('speedChartCanvas');
        
        if (!canvasContainer) {
            console.error('Canvas container not found');
            return;
        }
        
        // Check if we have data points
        if (!speedPoints || speedPoints.length === 0) {
            console.error('No speed data points to draw');
            canvasContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No speed data available to display</div>';
            return;
        }
        
        console.log(`Drawing chart with ${speedPoints.length} speed points`);
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        
        // Set canvas size, use fixed size if container size is 0
        const containerWidth = canvasContainer.offsetWidth || 800;
        const containerHeight = canvasContainer.offsetHeight || 400;
        
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        
        console.log(`Canvas size: ${canvas.width} x ${canvas.height}`);
        
        canvasContainer.innerHTML = '';
        canvasContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const padding = 60;
        const chartWidth = canvas.width - 2 * padding;
        const chartHeight = canvas.height - 2 * padding;
        
        // Prepare data - use collectiontime (Unix timestamp) for X-axis
        const collectionTimes = speedPoints.map(p => p.collectiontime);
        const metricKey = this.currentSelectedMetric || (this.currentSpeedData && this.currentSpeedData.metric) || 'speed';
        const values = speedPoints.map(p => {
            const n = Number(p[metricKey]);
            return Number.isFinite(n) ? n : 0;
        });
        
        // Check time data
        console.log('First few collection times:');
        speedPoints.slice(0, 3).forEach((p, i) => {
            const date = new Date(p.collectiontime);
            console.log(`  ${p.collectiontime} -> ${date} (${p.time_stamp})`);
        });
        
        const minTime = Math.min(...collectionTimes);
        const maxTime = Math.max(...collectionTimes);
        
        // Create time conversion function
        const timeToX = (timestamp) => {
            if (maxTime === minTime) return padding + chartWidth / 2;
            return padding + ((timestamp - minTime) / (maxTime - minTime)) * chartWidth;
        };
        
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        
        console.log(`Time range: ${new Date(minTime)} - ${new Date(maxTime)}`);
        
        // Add some padding to Y range
        const valuePadding = Math.max(1, (maxVal - minVal) * 0.1);
        const valueMin = minVal - valuePadding;
        const valueMax = maxVal + valuePadding;
        
        console.log(`Adjusted ${metricKey} range: ${valueMin} - ${valueMax}`);
        
        // Helper functions
        const valueToY = (val) => {
            if (valueMax === valueMin) return padding + chartHeight / 2;
            const normalized = (val - valueMin) / (valueMax - valueMin);
            return padding + chartHeight * (1 - normalized);
        };
        
        // Clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw traffic light periods in background
        if (trafficLightData && trafficLightData.traffic_lights.length > 0) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
            trafficLightData.traffic_lights.forEach(light => {
                const startTime = new Date(light.start_time);
                const endTime = new Date(light.end_time);
                
                // Only draw if the times are valid
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
        
        // Draw grid lines and labels
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        ctx.font = '12px Arial';
        ctx.fillStyle = '#666666';
        
        // Y-axis grid (metric)
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
        
        // X-axis grid (time)
        const timeSteps = 6;
        for (let i = 0; i <= timeSteps; i++) {
            const timestamp = minTime + (maxTime - minTime) * (i / timeSteps);
            const x = timeToX(timestamp);
            const time = new Date(timestamp);
            const timeStr = time.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
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
        
        // Group data by vehicle ID for multi-vehicle support
        const vehicleGroups = {};
        if (mode === 'single') {
            // Single vehicle mode - use all data as one group
            const vehicleId = speedPoints[0]?.vehicle_id || 1;
            vehicleGroups[vehicleId] = speedPoints;
        } else {
            // Time range mode - group by vehicle_id
            speedPoints.forEach(point => {
                if (!vehicleGroups[point.vehicle_id]) {
                    vehicleGroups[point.vehicle_id] = [];
                }
                vehicleGroups[point.vehicle_id].push(point);
            });
        }
        
        const vehicleColors = this.getVehicleColors(Object.keys(vehicleGroups).length);
        let colorIndex = 0;
        
        // Draw speed lines for each vehicle
        Object.keys(vehicleGroups).forEach(vehicleId => {
            const vehicleData = vehicleGroups[vehicleId];
            const vehicleColor = mode === 'single' ? 
                { red: 0.30, green: 0.69, blue: 0.31 } : // #4CAF50 for single mode
                vehicleColors[colorIndex];
            
            const colorStr = `rgb(${Math.floor(vehicleColor.red * 255)}, ${Math.floor(vehicleColor.green * 255)}, ${Math.floor(vehicleColor.blue * 255)})`;
            
            // Sort vehicle data by time
            vehicleData.sort((a, b) => a.collectiontime - b.collectiontime);
            
            // Draw line for this vehicle
            ctx.strokeStyle = colorStr;
            ctx.lineWidth = mode === 'single' ? 2 : 1.5;
            ctx.beginPath();
            
            let validPoints = 0;
            vehicleData.forEach((point, index) => {
                const x = timeToX(point.collectiontime);
                const y = valueToY(point[metricKey]);
                
                // Debug: log first few points of first vehicle
                if (colorIndex === 0 && index < 3) {
                    console.log(`Vehicle ${vehicleId} Point ${index}: time=${point.time_stamp}, ${metricKey}=${point[metricKey]}, collectiontime=${point.collectiontime}, x=${x}, y=${y}`);
                }
                
                // Check if coordinates are valid
                if (isNaN(x) || isNaN(y)) {
                    console.warn(`Invalid coordinates for vehicle ${vehicleId} point ${index}: x=${x}, y=${y}`);
                    return;
                }
                
                // Clamp coordinates to canvas bounds
                const clampedX = Math.max(padding, Math.min(canvas.width - padding, x));
                const clampedY = Math.max(padding, Math.min(canvas.height - padding, y));
                
                validPoints++;
                if (validPoints === 1) {
                    ctx.moveTo(clampedX, clampedY);
                } else {
                    ctx.lineTo(clampedX, clampedY);
                }
            });
            
            console.log(`Drawing line for vehicle ${vehicleId} with ${validPoints} valid points out of ${vehicleData.length} total points`);
            ctx.stroke();
            
            // Draw speed points for this vehicle
            ctx.fillStyle = colorStr;
            vehicleData.forEach((point) => {
                const x = timeToX(point.collectiontime);
                const y = valueToY(point[metricKey]);
                
                // Check if coordinates are valid
                if (isNaN(x) || isNaN(y)) {
                    return;
                }
                
                // Clamp coordinates to canvas bounds
                const clampedX = Math.max(padding, Math.min(canvas.width - padding, x));
                const clampedY = Math.max(padding, Math.min(canvas.height - padding, y));
                
                ctx.beginPath();
                ctx.arc(clampedX, clampedY, mode === 'single' ? 4 : 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            colorIndex++;
        });
        
        // Draw traffic light annotations
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
        
        // Draw axis labels
        ctx.fillStyle = '#333333';
        ctx.font = '14px Arial';
        ctx.fillText(`${metricKey}`, 10, 30);
        ctx.fillText('Time', canvas.width / 2 - 20, canvas.height - 10);
        
        // Draw border
        ctx.strokeStyle = '#cccccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(padding, padding, chartWidth, chartHeight);
    }

    /**
     * Show status message in speed analysis panel
     */
    showSpeedStatus(message, type = 'info') {
        const statusDiv = document.getElementById('speedStatus');
        const statusText = document.getElementById('speedStatusText');
        
        statusText.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        // Auto-hide after 3 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }

    /**
     * Load intersection topology from backend
     */
    async loadIntersectionTopology() {
        const roadId = document.getElementById('topologyRoadId').value;
        
        this.showTopologyStatus('Loading intersection topology...', 'info');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/topology/intersection?road_id=${roadId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.currentTopologyData = data;
                this.displayTopology(data.topology, data.summary);
                
                const featureCount = data.topology.features ? data.topology.features.length : 0;
                const armCount = data.summary ? data.summary.arms.length : 0;
                const edgeCount = data.summary ? data.summary.edge_supports.length : 0;
                
                this.updateTopologyInfo(data.summary);
                this.showTopologyStatus(
                    `Successfully loaded topology: ${armCount} arms, ${edgeCount} movement edges, ${featureCount} features`,
                    'success'
                );
                
                // Fly to the intersection center if available
                if (data.topology.features && data.topology.features.length > 0) {
                    const centerFeature = data.topology.features.find(f => f.properties.kind === 'center');
                    if (centerFeature && centerFeature.geometry.coordinates) {
                        const [lon, lat] = centerFeature.geometry.coordinates;
                        this.viewer.camera.flyTo({
                            destination: Cesium.Cartesian3.fromDegrees(lon, lat, 500),
                            duration: 2.0
                        });
                    }
                }
            } else {
                this.showTopologyStatus(`Error: ${data.message}`, 'error');
                if (data.hint) {
                    console.log('Hint:', data.hint);
                }
            }
        } catch (error) {
            console.error('Error loading topology:', error);
            this.showTopologyStatus('Failed to load topology data', 'error');
        }
    }

    /**
     * Display topology on the map
     */
    displayTopology(geojson, summary) {
        // Clear existing topology
        this.topologyDataSource.entities.removeAll();
        
        if (!geojson || !geojson.features) {
            console.warn('No features in topology data');
            return;
        }
        
        console.log(`Displaying ${geojson.features.length} topology features`);
        
        geojson.features.forEach(feature => {
            const props = feature.properties || {};
            const geom = feature.geometry;
            
            if (props.kind === 'center') {
                // Display intersection center as a large point
                const [lon, lat] = geom.coordinates;
                this.topologyDataSource.entities.add({
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
                // Display arm nodes
                const [lon, lat] = geom.coordinates;
                const armId = props.id;
                const angleRad = props.angle_rad;
                const angleDeg = (angleRad * 180 / Math.PI).toFixed(1);
                
                this.topologyDataSource.entities.add({
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
                // Display movement edges as polylines
                const coords = geom.coordinates;
                const positions = coords.map(coord => Cesium.Cartesian3.fromDegrees(coord[0], coord[1]));
                const weight = props.weight || 1;
                
                // Color based on weight (traffic volume)
                let color = Cesium.Color.GREEN;
                if (weight > 15) {
                    color = Cesium.Color.RED;
                } else if (weight > 10) {
                    color = Cesium.Color.ORANGE;
                } else if (weight > 5) {
                    color = Cesium.Color.YELLOW;
                }
                
                this.topologyDataSource.entities.add({
                    polyline: {
                        positions: positions,
                        width: Math.min(2 + weight * 0.5, 10),
                        material: color.withAlpha(0.8),
                        clampToGround: true
                    },
                    description: `<h3>Movement Edge</h3><p>From Arm ${props.u} to Arm ${props.v}</p><p>Traffic Volume: ${weight} vehicles</p>`
                });
                
                // Add arrow at the end to show direction
                if (coords.length >= 2) {
                    const lastCoord = coords[coords.length - 1];
                    const secondLastCoord = coords[coords.length - 2];
                    
                    // Calculate arrow direction
                    const dx = lastCoord[0] - secondLastCoord[0];
                    const dy = lastCoord[1] - secondLastCoord[1];
                    const heading = Math.atan2(dx, dy);
                    
                    this.topologyDataSource.entities.add({
                        position: Cesium.Cartesian3.fromDegrees(lastCoord[0], lastCoord[1]),
                        billboard: {
                            image: this.createArrowCanvas(color),
                            scale: 0.5,
                            rotation: -heading,
                            alignedAxis: Cesium.Cartesian3.UNIT_Z
                        }
                    });
                }
            }
        });
        
        console.log('Topology display completed');
    }

    /**
     * Create arrow canvas for direction indication
     */
    createArrowCanvas(color) {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        
        // Draw arrow
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

    /**
     * Update topology info display
     */
    updateTopologyInfo(summary) {
        const infoDiv = document.getElementById('topologyInfo');
        
        if (!summary) {
            infoDiv.innerHTML = '<div class="status-message">Topology loaded but no summary available.</div>';
            return;
        }
        
        let html = '<div class="topology-details">';
        html += `<h4>Topology Summary</h4>`;
        html += `<p><strong>Road Arms:</strong> ${summary.arms.length}</p>`;
        html += `<ul>`;
        summary.arms.forEach(arm => {
            html += `<li>Arm ${arm.id}: ${arm.angle_deg.toFixed(1)}°</li>`;
        });
        html += `</ul>`;
        html += `<p><strong>Movement Edges:</strong> ${summary.edge_supports.length}</p>`;
        html += `<ul>`;
        summary.edge_supports.forEach(edge => {
            html += `<li>Arm ${edge.u} → Arm ${edge.v}: ${edge.weight} vehicles</li>`;
        });
        html += `</ul>`;
        html += `<p><strong>Total Movements:</strong> ${summary.movements_total}</p>`;
        html += '</div>';
        
        infoDiv.innerHTML = html;
    }

    /**
     * Clear topology from map
     */
    clearTopology() {
        this.topologyDataSource.entities.removeAll();
        this.currentTopologyData = null;
        
        const infoDiv = document.getElementById('topologyInfo');
        infoDiv.innerHTML = '<div class="status-message">Topology cleared. Click "Load Road Network" to display again.</div>';
        
        this.showTopologyStatus('Topology cleared', 'success');
    }

    /**
     * Show status message in topology panel
     */
    showTopologyStatus(message, type = 'info') {
        const statusDiv = document.getElementById('topologyStatus');
        const statusText = document.getElementById('topologyStatusText');
        
        statusText.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        // Auto-hide after 3 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }
}

// Initialize application when page loading is complete
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing geographic data visualization application...');
    
    // Ensure modals are hidden immediately
    const timelineModal = document.getElementById('timelineModal');
    const speedModal = document.getElementById('speedModal');
    
    if (timelineModal) {
        timelineModal.style.display = 'none';
        timelineModal.style.visibility = 'hidden';
    }
    
    if (speedModal) {
        speedModal.style.display = 'none';
        speedModal.style.visibility = 'hidden';
    }
    
    try {
        const app = new GeoVisualization();
        window.geoApp = app; // Expose application instance to global scope for debugging
        console.log('Application initialization successful');
    } catch (error) {
        console.error('Application initialization failed:', error);
        alert('Application initialization failed, please check console error messages');
    }
});
