// Geographic Data Visualization Application Main File

class GeoVisualization {
    constructor() {
        this.viewer = null;
        this.dataSource = null;
        this.currentVehicleData = [];
        this.apiBaseUrl = 'http://127.0.0.1:5000/api';
        
        this.init();
        this.bindEvents();
    }

    /**
     * Initialize Cesium map
     */
    init() {
        // Configure Cesium access token (if needed)
        // Cesium.Ion.defaultAccessToken = 'your_access_token_here';

        // Create Cesium viewer
        this.viewer = new Cesium.Viewer('cesiumContainer', {
            terrainProvider: Cesium.createWorldTerrain(),
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

        loadDataBtn.addEventListener('click', () => {
            this.loadVehicleData();
        });

        clearDataBtn.addEventListener('click', () => {
            this.clearData();
        });
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

            const vehicleId = document.getElementById('vehicleId').value;
            const dataLimit = document.getElementById('dataLimit').value;

            // Build API URL
            let apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_id=${vehicleId}`;
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
                const summaryResponse = await fetch(`${this.apiBaseUrl}/vehicle/summary?vehicle_id=${vehicleId}`);
                const summaryResult = await summaryResponse.json();
                
                // Visualize data
                this.visualizeData(this.currentVehicleData);
                this.updateDataInfo(summaryResult);
                
                this.showStatus(`Successfully loaded ${result.total_points} data points`);
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

        const positions = [];
        const pointEntities = [];

        // Create entity for each data point
        data.forEach((point, index) => {
            const position = Cesium.Cartesian3.fromDegrees(
                point.longitude, 
                point.latitude, 
                0
            );

            positions.push(position);

            // Determine point color based on speed
            let color = this.getSpeedColor(point.speed);

            // Create point entity
            const pointEntity = this.dataSource.entities.add({
                position: position,
                point: {
                    pixelSize: 8,
                    color: color,
                    outlineColor: Cesium.Color.WHITE,
                    outlineWidth: 2,
                    heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
                },
                description: this.createPointDescription(point, index)
            });

            pointEntities.push(pointEntity);
        });

        // Create trajectory line
        if (positions.length > 1) {
            this.dataSource.entities.add({
                polyline: {
                    positions: positions,
                    width: 3,
                    material: Cesium.Color.CYAN.withAlpha(0.7),
                    clampToGround: true
                }
            });
        }

        // Fly to data range
        this.viewer.flyTo(this.dataSource);

        console.log(`Visualized ${data.length} data points`);
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
    createPointDescription(point, index) {
        return `
            <div class="point-info">
                <h4>Data Point #${index + 1}</h4>
                <table>
                    <tr><td>Vehicle ID:</td><td>${point.vehicle_id}</td></tr>
                    <tr><td>Time:</td><td>${point.time_stamp}</td></tr>
                    <tr><td>Road ID:</td><td>${point.road_id}</td></tr>
                    <tr><td>Longitude:</td><td>${point.longitude.toFixed(6)}</td></tr>
                    <tr><td>Latitude:</td><td>${point.latitude.toFixed(6)}</td></tr>
                    <tr><td>Speed:</td><td>${point.speed.toFixed(1)} km/h</td></tr>
                    <tr><td>Accelerator:</td><td>${point.acceleratorpedal}%</td></tr>
                    <tr><td>Brake Status:</td><td>${point.brakestatus ? 'Braking' : 'Normal'}</td></tr>
                </table>
            </div>
        `;
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
