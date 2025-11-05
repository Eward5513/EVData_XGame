// Geographic Data Visualization Application Main File

import { API_BASE_URL, ANALYSIS_CENTERS, CENTER_CHANGE_RADIUS_METERS } from './config.js';
import { getSpeedColor as getSpeedColorUtil, getVehicleColors as getVehicleColorsUtil, blendColors as blendColorsUtil } from './utils/color.js';
import { offsetLonLat } from './utils/geo.js';
import { drawAnalysisCenterOverlay as drawCenterOverlay } from './overlays/analysisOverlay.js';
import { drawIntersectionInference as drawInference } from './overlays/inference.js';
import { displayTopology as displayTopologyExt } from './overlays/topology.js';
import { drawSpeedChart as drawSpeedChartExt } from './charts/speedChart.js';
import { drawGenericMetricChart as drawGenericMetricChartExt } from './charts/genericChart.js';
import { renderVehicleTrajectories } from './renderers/vehicle.js';
import { renderExcludedTrajectories } from './renderers/excluded.js';

class GeoVisualization {
    constructor() {
        this.viewer = null;
        this.dataSource = null;
        this.topologyDataSource = null;
        this.overlayDataSource = null;
        this.excludedDataSource = null;
        this.currentVehicleData = [];
        this.currentTrafficLightData = null;
        this.currentSpeedData = null;
        this.currentSpeedTrafficLights = null;
        this.currentTopologyData = null;
        this.apiBaseUrl = API_BASE_URL;
        this.currentSelectedMetric = 'speed';
        this.currentSelectedSegId = null;
        this.currentVehicleFilterMode = 'direction';
        this.currentExcludedData = [];
        this.allDates = [];
        this.allRoadIds = [];
        
		// Analysis centers and radius (A0003 and A0008 should both be displayed)
		this.analysisCenters = ANALYSIS_CENTERS;
        this.centerChangeRadiusMeters = CENTER_CHANGE_RADIUS_METERS;
        
        this.bindEvents();
        this.init();
    }

    /**
     * Initialize Cesium map
     */
    async init() {
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

        this.excludedDataSource = new Cesium.CustomDataSource('excludedTrack');
        this.viewer.dataSources.add(this.excludedDataSource);

        // Set initial view to China region
        this.viewer.camera.setView({
            destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 1000000)
        });

        // Preload road IDs and populate all road selects
        try {
            await this.preloadAllRoadIds();
            this.populateAllRoadSelects(this.allRoadIds);
        } catch (e) {
            console.warn('Failed to preload road IDs:', e);
        }

        // Populate raw dates for default road in Raw panel
        try {
            await this.loadRawDates();
        } catch (e) {
            // ignore errors for raw dates preload
        }

        // Preload all dates (union across roads) and populate date selects
        this.allDates = [];
        try {
            await this.preloadAllDates();
            this.populateAllDateSelects(this.allDates);
        } catch (e) {
            console.warn('Failed to preload dates:', e);
        }

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
        if (!this.overlayDataSource) return;
        drawCenterOverlay(this.overlayDataSource, this.analysisCenters, this.centerChangeRadiusMeters);
    }

    /**
     * Draw a simple line chart for a generic metric over time using canvas
     */
    drawGenericMetricChart(containerEl, points, metricKey, title = '') {
        return drawGenericMetricChartExt(containerEl, points, metricKey, title);
    }

    /**
     * Load and render intersection inference overlays for configured roads
     */
    async loadIntersectionInferenceOverlays() {
        const roadIds = ['A0003', 'A0008'];
        try {
            const tasks = roadIds.map(async (rid) => {
                const url = `${this.apiBaseUrl}/intersection/inference?road_id=${rid}`;
                const resp = await fetch(url);
                const data = await resp.json();
                if (data.status === 'success' && data.inference) {
                    this.drawIntersectionInference(rid, data.inference);
                } else {
                    console.warn(`Failed to load intersection inference for ${rid}:`, data.message);
                }
            });
            await Promise.all(tasks);
        } catch (e) {
            console.error('Error loading intersection inference overlays:', e);
        }
    }

    clearIntersectionInferenceOverlays() {
        if (this.topologyDataSource) {
            this.topologyDataSource.entities.removeAll();
        }
    }

    

    /**
     * Draw center, axes and stop lines from inference JSON for a given road
     */
    drawIntersectionInference(roadId, inference) {
        if (!this.topologyDataSource) return;
        drawInference(this.topologyDataSource, roadId, inference, offsetLonLat);
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        const loadDataBtn = document.getElementById('loadDataBtn');
        const clearDataBtn = document.getElementById('clearDataBtn');
        const pointsOnlyCheckbox = document.getElementById('pointsOnly');
        const vehicleModeSelect = document.getElementById('vehicleMode');
        const vehicleFilterModeSelect = null;

        // Excluded trajectories elements
        const loadExcludedBtn = document.getElementById('loadExcludedBtn');
        const clearExcludedBtn = document.getElementById('clearExcludedBtn');
        
        // Raw CSV panel elements
        const loadRawBtn = document.getElementById('loadRawBtn');
        const clearRawBtn = document.getElementById('clearRawBtn');
        const rawRoadSel = document.getElementById('rawRoadId');
        
        // Traffic light query elements
        const loadCyclesBtn = document.getElementById('loadCyclesBtn');
        const queryTrafficLightBtn = document.getElementById('queryTrafficLightBtn');
        const closeTimelineModal = document.getElementById('closeTimelineModal');
        const timelineModal = document.getElementById('timelineModal');
        
        // Speed analysis elements
        const analyzeSpeedBtn = document.getElementById('analyzeSpeedBtn');
        const speedModeSelect = document.getElementById('speedMode');
        const loadTimeRangeBtn = document.getElementById('loadTimeRangeBtn');
        const closeSpeedModal = document.getElementById('closeSpeedModal');
        const speedModal = document.getElementById('speedModal');
        const speedMetricSelect = document.getElementById('speedMetric');
        const speedSegIdInput = document.getElementById('speedSegId');
        
        // Topology elements (panel removed) - guard for null
        const loadTopologyBtn = document.getElementById('loadTopologyBtn');
        const clearTopologyBtn = document.getElementById('clearTopologyBtn');

        // Intersection overlay (inference) elements
        const loadInferenceBtn = document.getElementById('loadInferenceBtn');
        const clearInferenceBtn = document.getElementById('clearInferenceBtn');


        loadDataBtn.addEventListener('click', () => {
            this.loadVehicleData();
        });

        clearDataBtn.addEventListener('click', () => {
            this.clearData();
        });

        // Re-render when toggling points-only if data is already loaded
        if (pointsOnlyCheckbox) {
            pointsOnlyCheckbox.addEventListener('change', () => {
                if (this.currentVehicleData && this.currentVehicleData.length > 0) {
                    this.visualizeData(this.currentVehicleData);
                }
            });
        }

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

        // Road selection change needs no handler (dates are preloaded globally)

        // Movement filter mode removed; always use direction filters
        this.currentVehicleFilterMode = 'direction';

        

        // Excluded panel events
        // No per-road excluded date loading; dates are preloaded globally
        if (loadExcludedBtn) {
            loadExcludedBtn.addEventListener('click', () => {
                this.loadExcludedData();
            });
        }
        if (clearExcludedBtn) {
            clearExcludedBtn.addEventListener('click', () => {
                this.clearExcludedData();
            });
        }

        // Raw CSV panel events
        if (loadRawBtn) {
            loadRawBtn.addEventListener('click', () => {
                this.loadRawData();
            });
        }
        if (clearRawBtn) {
            clearRawBtn.addEventListener('click', () => {
                this.clearRawData();
            });
        }
        if (rawRoadSel) {
            rawRoadSel.addEventListener('change', () => {
                this.loadRawDates();
            });
        }

        // Traffic light query event listeners
        loadCyclesBtn.addEventListener('click', () => {
            this.loadAvailableCycles();
        });

        queryTrafficLightBtn.addEventListener('click', () => {
            this.queryTrafficLightStatus();
        });

        // Speed analysis event listeners (no date loading button)

        analyzeSpeedBtn.addEventListener('click', () => {
            this.analyzeSpeed();
        });

        // Speed road selection change needs no handler (dates are preloaded globally)

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

        // Topology event listeners (panel removed)
        if (loadTopologyBtn) {
            loadTopologyBtn.addEventListener('click', () => {
                this.loadIntersectionTopology();
            });
        }
        if (clearTopologyBtn) {
            clearTopologyBtn.addEventListener('click', () => {
                this.clearTopology();
            });
        }

        if (loadInferenceBtn) {
            loadInferenceBtn.addEventListener('click', async () => {
                try {
                    const status = document.getElementById('inferenceStatus');
                    const text = document.getElementById('inferenceStatusText');
                    if (status && text) { status.style.display = 'block'; text.textContent = 'Loading overlays...'; }
                    await this.loadIntersectionInferenceOverlays();
                    if (status && text) { text.textContent = 'Overlays loaded'; setTimeout(()=> status.style.display='none', 2000); }
                } catch (e) {
                    const status = document.getElementById('inferenceStatus');
                    const text = document.getElementById('inferenceStatusText');
                    if (status && text) { status.style.display = 'block'; text.textContent = `Error: ${e.message}`; }
                }
            });
        }
        if (clearInferenceBtn) {
            clearInferenceBtn.addEventListener('click', () => {
                this.clearIntersectionInferenceOverlays();
                const status = document.getElementById('inferenceStatus');
                const text = document.getElementById('inferenceStatusText');
                if (status && text) { status.style.display = 'block'; text.textContent = 'Overlays cleared'; setTimeout(()=> status.style.display='none', 1500); }
            });
        }

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

        this.updateVehicleFilterVisibility();
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

        this.updateVehicleFilterVisibility();
    }

    updateVehicleFilterVisibility() {
        const directionContainers = document.querySelectorAll('.direction-filter-container');
        directionContainers.forEach((el) => {
            if (el) { el.style.display = 'block'; }
        });
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
            const dataSourceSel = document.getElementById('dataSource');
            const src = (dataSourceSel && dataSourceSel.value) || 'merged';
            const timeRangeInfo = document.getElementById('vehicleTimeRangeInfo');
            
            if (!date) {
                this.showStatus('Please select a date first');
                return;
            }
            
            timeRangeInfo.textContent = 'Loading time range...';
            
            const response = await fetch(`${this.apiBaseUrl}/speed/time-range?road_id=${roadId}&date=${date}&source=${encodeURIComponent(src)}`);
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
     * Preload all available dates (union across all configured roads)
     */
    async preloadAllDates() {
        try {
            const roadIds = (this.allRoadIds && this.allRoadIds.length > 0)
                ? this.allRoadIds
                : Object.keys(this.analysisCenters || {});
            const union = new Set();
            const sourceSel = document.getElementById('dataSource');
            const sourceVal = (sourceSel && sourceSel.value) || 'merged';
            for (const r of roadIds) {
                try {
                    const response = await fetch(`${this.apiBaseUrl}/vehicle/dates?road_id=${r}&source=${encodeURIComponent(sourceVal)}`);
                    if (!response.ok) continue;
                    const result = await response.json();
                    if (result && result.status === 'success' && Array.isArray(result.dates)) {
                        result.dates.forEach(d => union.add(String(d)));
                    }
                } catch (e) {
                    // ignore per-road errors
                }
            }
            this.allDates = Array.from(union).sort();
        } catch (e) {
            this.allDates = [];
            throw e;
        }
    }

    /**
     * Populate all date selects with the same list
     */
    populateAllDateSelects(dates) {
        const ids = ['dateFilter', 'speedDate', 'excludedDate'];
        ids.forEach(id => {
            const sel = document.getElementById(id);
            if (!sel) return;
            sel.innerHTML = '<option value="" selected>All Dates</option>';
            (dates || []).forEach(d => {
                const opt = document.createElement('option');
                opt.value = String(d);
                opt.textContent = String(d);
                sel.appendChild(opt);
            });
        });
    }

    /**
     * Preload road IDs from configuration (analysisCenters)
     */
    async preloadAllRoadIds() {
        try {
            const centers = this.analysisCenters || {};
            this.allRoadIds = Object.keys(centers);
        } catch (e) {
            this.allRoadIds = [];
            throw e;
        }
    }

    /**
     * Populate all road selects with the same list
     */
    populateAllRoadSelects(roadIds) {
        const ids = ['roadId', 'trafficRoadId', 'speedRoadId', 'excludedRoadId', 'topologyRoadId', 'rawRoadId'];
        ids.forEach(id => {
            const sel = document.getElementById(id);
            if (!sel) return;
            sel.innerHTML = '';
            (roadIds || []).forEach(r => {
                const opt = document.createElement('option');
                opt.value = String(r);
                opt.textContent = String(r);
                sel.appendChild(opt);
            });
        });
    }

    /**
     * Excluded panel: status helper
     */
    showExcludedStatus(message, visible = true) {
        const status = document.getElementById('excludedStatus');
        const text = document.getElementById('excludedStatusText');
        if (status && text) {
            text.textContent = message || '';
            status.style.display = visible ? 'block' : 'none';
        }
    }

    /**
     * Raw panel: status helper
     */
    showRawStatus(message, visible = true) {
        const status = document.getElementById('rawStatus');
        const text = document.getElementById('rawStatusText');
        if (status && text) {
            text.textContent = message || '';
            status.style.display = visible ? 'block' : 'none';
        }
    }

    /**
     * Load available raw dates for selected road
     */
    async loadRawDates() {
        const roadSel = document.getElementById('rawRoadId');
        const dateSel = document.getElementById('rawDate');
        if (!roadSel || !dateSel) return;
        const roadId = roadSel.value;
        try {
            this.showRawStatus('Loading dates...', true);
            const resp = await fetch(`${this.apiBaseUrl}/raw/dates?road_id=${roadId}`);
            const result = await resp.json();
            dateSel.innerHTML = '<option value="" selected>Select Date</option>';
            if (result && result.status === 'success' && Array.isArray(result.dates)) {
                result.dates.forEach(d => {
                    const opt = document.createElement('option');
                    opt.value = String(d);
                    opt.textContent = String(d);
                    dateSel.appendChild(opt);
                });
                this.showRawStatus(`Loaded ${result.total_dates} dates`, true);
                setTimeout(()=> this.showRawStatus('', false), 1500);
            } else {
                this.showRawStatus('No dates available', true);
            }
        } catch (e) {
            this.showRawStatus(`Failed to load dates: ${e.message}`, true);
        }
    }

    /**
     * Load raw CSV trajectory data
     */
    async loadRawData() {
        try {
            const roadId = document.getElementById('rawRoadId').value;
            const date = document.getElementById('rawDate').value;
            const vehicleId = document.getElementById('rawVehicleId') ? document.getElementById('rawVehicleId').value : '';

            if (!date) {
                this.showRawStatus('Please select a date', true);
                return;
            }

            this.showRawStatus('Loading raw trajectories...', true);

            let url = `${this.apiBaseUrl}/raw/data?road_id=${roadId}&date=${encodeURIComponent(date)}`;
            if (vehicleId) url += `&vehicle_id=${encodeURIComponent(vehicleId)}`;

            const response = await fetch(url);
            const result = await response.json();
            if (result.status === 'success') {
                const data = result.data || [];
                this.currentVehicleData = data;
                this.visualizeData(data);
                this.showRawStatus(`Loaded ${data.length} points`);
            } else {
                throw new Error(result.message || 'Failed to load raw data');
            }
        } catch (e) {
            console.error('Error loading raw data:', e);
            this.showRawStatus(`Failed to load raw data: ${e.message}`, true);
        }
    }

    /**
     * Clear raw visualization (reuses main clear)
     */
    clearRawData() {
        this.clearData();
        this.showRawStatus('Cleared', true);
        setTimeout(()=> this.showRawStatus('', false), 1200);
    }

    

    /**
     * Load excluded trajectory points for selected road/date
     */
    async loadExcludedData() {
        try {
            const roadId = document.getElementById('excludedRoadId').value;
            const date = document.getElementById('excludedDate').value;
            const vehicleId = document.getElementById('excludedVehicleId') ? document.getElementById('excludedVehicleId').value : '';
            this.showExcludedStatus('Loading excluded trajectories...', true);

            let url = `${this.apiBaseUrl}/excluded/data?road_id=${roadId}`;
            if (date) {
                url += `&date=${encodeURIComponent(date)}`;
            }
            if (vehicleId) {
                url += `&vehicle_id=${encodeURIComponent(vehicleId)}`;
            }

            const response = await fetch(url);
            const result = await response.json();
            if (result.status === 'success') {
                const data = result.data || [];
                this.currentExcludedData = data;
                this.visualizeExcludedData(data);
                const segments = result.total_segments != null ? result.total_segments : 'N/A';
                this.showExcludedStatus(`Loaded ${data.length} points from ${segments} segment(s)`);
            } else {
                throw new Error(result.message || 'Failed to load excluded data');
            }
        } catch (e) {
            console.error('Error loading excluded data:', e);
            this.showExcludedStatus(`Failed to load excluded data: ${e.message}`);
        }
    }

    /**
     * Clear excluded visualization
     */
    clearExcludedData() {
        if (this.excludedDataSource) {
            this.excludedDataSource.entities.removeAll();
        }
        this.currentExcludedData = [];
        this.showExcludedStatus('Cleared', true);
    }

    /**
     * Visualize excluded trajectories with distinct style
     */
    visualizeExcludedData(data) {
        if (!this.excludedDataSource || !this.viewer || !Array.isArray(data) || data.length === 0) return;
        renderExcludedTrajectories(
            this.excludedDataSource,
            this.viewer,
            data,
            (s) => this.getSpeedColor(s),
            (n) => this.getVehicleColors(n),
            (a, b) => this.blendColors(a, b),
            (p, i, multi, d) => this.createPointDescription(p, i, multi, d)
        );
    }

    /**
     * Load vehicle data from backend
     */
    async loadVehicleData() {
        try {
            this.showStatus('Loading data...', true);

            const vehicleMode = document.getElementById('vehicleMode').value;
            const roadId = document.getElementById('roadId').value;
            const dataSource = (document.getElementById('dataSource') && document.getElementById('dataSource').value) || 'merged';
            const dateFilter = document.getElementById('dateFilter').value;
            const filterMode = this.currentVehicleFilterMode || 'direction';

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
                apiUrl = `${this.apiBaseUrl}/vehicle/data?vehicle_count=${vehicleCount}&road_id=${roadId}`;
                summaryUrl = `${this.apiBaseUrl}/vehicle/summary?vehicle_count=${vehicleCount}&road_id=${roadId}`;
                
                {
                    const selected = Array.from(document.querySelectorAll('input[name="batchDirection"]:checked')).map(el => el.value);
                    if (selected.length > 0) {
                        const dirParam = selected.join(',');
                        apiUrl += `&direction=${encodeURIComponent(dirParam)}`;
                        summaryUrl += `&direction=${encodeURIComponent(dirParam)}`;
                    }
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
                
                {
                    const selected = Array.from(document.querySelectorAll('input[name="timeRangeDirection"]:checked')).map(el => el.value);
                    if (selected.length > 0) {
                        const dirParam = selected.join(',');
                        apiUrl += `&direction=${encodeURIComponent(dirParam)}`;
                        summaryUrl += `&direction=${encodeURIComponent(dirParam)}`;
                    }
                }
            }

            // Add date filter if specified
            if (dateFilter) {
                apiUrl += `&date=${dateFilter}`;
                summaryUrl += `&date=${dateFilter}`;
            }

            // Add source selection
            if (dataSource) {
                apiUrl += `&source=${encodeURIComponent(dataSource)}`;
                summaryUrl += `&source=${encodeURIComponent(dataSource)}`;
            }

            // Single request path (backend supports multi-direction via comma-separated param)
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
        if (!this.dataSource || !this.viewer || !data || data.length === 0) {
            if (!data || data.length === 0) console.warn('No data to visualize');
            return;
        }
        const pointsOnly = !!(document.getElementById('pointsOnly') && document.getElementById('pointsOnly').checked);
        renderVehicleTrajectories(
            this.dataSource,
            this.viewer,
            data,
            (s) => this.getSpeedColor(s),
            (n) => this.getVehicleColors(n),
            (a, b) => this.blendColors(a, b),
            (p, i, multi, d) => this.createPointDescription(p, i, multi, d),
            pointsOnly
        );
    }

    /**
     * Get color based on speed
     */
    getSpeedColor(speed) {
        return getSpeedColorUtil(speed);
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
                    ${point.end_time ? `<tr><td>End Time:</td><td>${point.end_time}</td></tr>` : ''}
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
        return getVehicleColorsUtil(vehicleCount);
    }

    /**
     * Blend two colors together
     */
    blendColors(vehicleColor, speedColor, ratio = 0.7) {
        return blendColorsUtil(vehicleColor, speedColor, ratio);
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
            
            const dsSel = document.getElementById('dataSource');
            const ds = (dsSel && dsSel.value) || 'merged';
            const response = await fetch(`${this.apiBaseUrl}/speed/time-range?road_id=${roadId}&date=${date}&source=${encodeURIComponent(ds)}`);
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
     * Draw speed chart using canvas (delegated to chart module)
     */
    drawSpeedChart(speedPoints, trafficLightData, mode = 'single', vehicleIds = []) {
        const metricKey = this.currentSelectedMetric || (this.currentSpeedData && this.currentSpeedData.metric) || 'speed';
        return drawSpeedChartExt(speedPoints, trafficLightData, mode, vehicleIds, metricKey);
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
     * Display topology on the map (delegated to overlay module)
     */
    displayTopology(geojson, summary) {
        return displayTopologyExt(geojson, summary, this.topologyDataSource, this.viewer);
    }

    /**
     * Create arrow canvas for direction indication
     */
    // Removed: now provided by overlays/topology utilities

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
