// Geographic Data Visualization Application Main File

class GeoVisualization {
    constructor() {
        this.viewer = null;
        this.dataSource = null;
        this.currentVehicleData = [];
        this.apiBaseUrl = 'http://127.0.0.1:5000/api';
        
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

        // Create data source
        this.dataSource = new Cesium.CustomDataSource('vehicleTrack');
        this.viewer.dataSources.add(this.dataSource);

        // Set initial view to China region
        this.viewer.camera.setView({
            destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 1000000)
        });

        console.log('Cesium map initialization completed');
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        const loadDataBtn = document.getElementById('loadDataBtn');
        const clearDataBtn = document.getElementById('clearDataBtn');
        const vehicleModeSelect = document.getElementById('vehicleMode');

        loadDataBtn.addEventListener('click', () => {
            this.loadVehicleData();
        });

        clearDataBtn.addEventListener('click', () => {
            this.clearData();
        });

        // Handle vehicle mode switching
        vehicleModeSelect.addEventListener('change', (e) => {
            this.toggleVehicleMode(e.target.value);
        });
    }

    /**
     * Toggle between single and batch vehicle selection modes
     */
    toggleVehicleMode(mode) {
        const singleSection = document.getElementById('singleVehicleSection');
        const batchSection = document.getElementById('batchVehicleSection');

        if (mode === 'single') {
            singleSection.style.display = 'block';
            batchSection.style.display = 'none';
        } else {
            singleSection.style.display = 'none';
            batchSection.style.display = 'block';
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
     * Load vehicle data from backend
     */
    async loadVehicleData() {
        try {
            this.showStatus('Loading data...', true);

            const vehicleMode = document.getElementById('vehicleMode').value;
            const roadId = document.getElementById('roadId').value;
            const dataLimit = document.getElementById('dataLimit').value;

            let apiUrl;
            let summaryUrl;

            if (vehicleMode === 'single') {
                // Single vehicle mode
                const vehicleId = document.getElementById('vehicleId').value;
                apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_id=${vehicleId}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?vehicle_id=${vehicleId}&road_id=${roadId}`;
            } else {
                // Batch vehicle mode
                const vehicleCount = document.getElementById('vehicleCount').value;
                apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_count=${vehicleCount}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?vehicle_count=${vehicleCount}&road_id=${roadId}`;
            }

            if (dataLimit) {
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
                this.showStatus(`Successfully loaded ${totalPoints} data points from ${vehicleMode === 'single' ? '1 vehicle' : result.vehicle_count + ' vehicles'}`);
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

        // Group data by vehicle ID
        const vehicleGroups = {};
        data.forEach(point => {
            const vehicleId = point.vehicle_id;
            if (!vehicleGroups[vehicleId]) {
                vehicleGroups[vehicleId] = [];
            }
            vehicleGroups[vehicleId].push(point);
        });

        const vehicleIds = Object.keys(vehicleGroups);
        const colors = this.getVehicleColors(vehicleIds.length);

        // Process each vehicle separately
        vehicleIds.forEach((vehicleId, vehicleIndex) => {
            const vehicleData = vehicleGroups[vehicleId];
            const vehicleColor = colors[vehicleIndex];
            const positions = [];

            // Create entity for each data point of this vehicle
            vehicleData.forEach((point, index) => {
                const position = Cesium.Cartesian3.fromDegrees(
                    point.longitude, 
                    point.latitude, 
                    0
                );

                positions.push(position);

                // Determine point color: use vehicle color modulated by speed
                let pointColor = this.getSpeedColor(point.speed);
                if (vehicleIds.length > 1) {
                    // In multi-vehicle mode, blend speed color with vehicle color
                    pointColor = this.blendColors(vehicleColor, pointColor);
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
                    description: this.createPointDescription(point, index, vehicleIds.length > 1)
                });
            });

            // Create trajectory line for this vehicle
            if (positions.length > 1) {
                this.dataSource.entities.add({
                    polyline: {
                        positions: positions,
                        width: vehicleIds.length > 1 ? 4 : 3,
                        material: vehicleColor.withAlpha(0.8),
                        clampToGround: true
                    },
                    description: `Vehicle ${vehicleId} trajectory (${vehicleData.length} points)`
                });
            }
        });

        // Fly to data range
        this.viewer.flyTo(this.dataSource);

        console.log(`Visualized ${data.length} data points from ${vehicleIds.length} vehicle(s)`);
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
    createPointDescription(point, index, isMultiVehicle = false) {
        const title = isMultiVehicle ? 
            `Vehicle ${point.vehicle_id} - Point #${index + 1}` : 
            `Data Point #${index + 1}`;
            
        return `
            <div class="point-info">
                <h4>${title}</h4>
                <table>
                    <tr><td>Vehicle ID:</td><td>${point.vehicle_id}</td></tr>
                    <tr><td>Time:</td><td>${point.time_stamp}</td></tr>
                    <tr><td>Road ID:</td><td>${point.road_id}</td></tr>
                    <tr><td>Longitude:</td><td>${point.longitude.toFixed(6)}</td></tr>
                    <tr><td>Latitude:</td><td>${point.latitude.toFixed(6)}</td></tr>
                    <tr><td>Speed:</td><td>${point.speed.toFixed(1)} km/h</td></tr>
                    <tr><td>Accelerator:</td><td>${point.acceleratorpedal}%</td></tr>
                    <tr><td>Brake Status:</td><td>${point.brakestatus}</td></tr>
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
}

// Initialize application when page loading is complete
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing geographic data visualization application...');
    
    try {
        const app = new GeoVisualization();
        window.geoApp = app; // Expose application instance to global scope for debugging
        console.log('Application initialization successful');
    } catch (error) {
        console.error('Application initialization failed:', error);
        alert('Application initialization failed, please check console error messages');
    }
});
