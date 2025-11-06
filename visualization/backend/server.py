#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs

# CSV/data base path moved under repository data directory
CSV_BASE_PATH = '../../data'
TRAFFIC_SIGNAL_FILE = os.path.join(CSV_BASE_PATH, 'traffic_signal.csv')
DIRECTION_FILE = os.path.join(CSV_BASE_PATH, 'direction.csv')

# Global variable to cache direction data
direction_data = None

def load_direction_data():
    """Load direction data from direction.csv"""
    global direction_data
    if direction_data is None:
        try:
            direction_data = pd.read_csv(DIRECTION_FILE)
            print(f"Loaded direction data: {len(direction_data)} records")
        except FileNotFoundError:
            print(f"Direction file not found: {DIRECTION_FILE}")
            direction_data = pd.DataFrame()  # Empty dataframe as fallback
        except Exception as e:
            print(f"Error loading direction data: {e}")
            direction_data = pd.DataFrame()
    return direction_data

def _prefer_merged_csv_path(road_id: str) -> str:
    """Return merged CSV path if exists, else fallback to split CSV."""
    merged = os.path.join(CSV_BASE_PATH, f'{road_id}_merged.csv')
    split = os.path.join(CSV_BASE_PATH, f'{road_id}_split.csv')
    return merged if os.path.exists(merged) else split

def _csv_path_for_source(road_id: str, source: str | None) -> str:
    """Resolve CSV path based on explicit source selection.
    source in {'split','original','before'} => split
    source in {'merged','after','merge'} => merged if exists else split
    source in {'refined','skeleton'} => refined if exists else merged if exists else split
    else => prefer merged
    """
    s = (str(source).strip().lower() if source is not None else '')
    refined = os.path.join(CSV_BASE_PATH, f'{road_id}_refined.csv')
    merged = os.path.join(CSV_BASE_PATH, f'{road_id}_merged.csv')
    split = os.path.join(CSV_BASE_PATH, f'{road_id}_split.csv')
    if s in ('split', 'original', 'before'):
        return split
    if s in ('merged', 'after', 'merge'):
        return merged if os.path.exists(merged) else split
    if s in ('refined', 'skeleton'):
        return refined if os.path.exists(refined) else (merged if os.path.exists(merged) else split)
    return merged if os.path.exists(merged) else split

def parse_directions_param(direction_value):
    """Parse 'direction' query parameter allowing single, comma-separated, or repeated values.
    Returns a list of uppercase direction codes (e.g., ['A1','B2']).
    """
    if direction_value is None:
        return []
    values = direction_value if isinstance(direction_value, list) else [direction_value]
    results = []
    for v in values:
        for part in str(v).split(','):
            code = part.strip().upper()
            if code:
                results.append(code)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for code in results:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return deduped

## Movement-based filters are deprecated; keep codebase lean by removing loader

def load_vehicle_data(vehicle_id=None,
                      road_id='A0003',
                      vehicle_count=None,
                      date=None,
                      start_time=None,
                      end_time=None,
                      direction=None,
                      source=None):
    """
    Load data for specified vehicle ID(s) and road ID using pandas
    
    Args:
        vehicle_id: Single vehicle ID to load (for single mode)
        road_id: Road intersection ID 
        vehicle_count: Number of vehicles to load from the beginning (for batch mode)
        date: Date filter in YYYY-MM-DD format (optional)
        start_time: Start time filter in HH:MM format (optional)
        end_time: End time filter in HH:MM format (optional)
        direction: Direction filter (e.g., A1-1/A1-2/B1-1/B1-2/A2-*/A3-* or C) (optional)
        source: Data source selector (split/merged/refined)
    """
    try:
        # Build CSV file path based on road_id and selected source
        csv_file_path = _csv_path_for_source(road_id, source)
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Filter by date if specified
        if date:
            df = df[df['date'] == date]
        
        # Filter by time range if specified
        if start_time and end_time:
            # Convert time_stamp to time for comparison
            df['time_only'] = pd.to_datetime(df['time_stamp']).dt.time
            start_time_obj = pd.to_datetime(start_time, format='%H:%M').time()
            end_time_obj = pd.to_datetime(end_time, format='%H:%M').time()
            df = df[(df['time_only'] >= start_time_obj) & (df['time_only'] <= end_time_obj)]
            df = df.drop('time_only', axis=1)  # Remove temporary column
        
        # Filter data based on mode
        if start_time and end_time:
            # Time range mode: get all vehicles in the time range
            vehicle_data = df
        elif vehicle_count is not None:
            # Batch mode: get vehicles with ID <= vehicle_count
            vehicle_data = df[df['vehicle_id'] <= vehicle_count]
        else:
            # Single mode: get specific vehicle (default to 1 if not specified)
            if vehicle_id is None:
                vehicle_id = 1
            vehicle_data = df[df['vehicle_id'] == vehicle_id]
        
        # Apply direction filter if specified (single or multiple)
        if direction:
            directions = parse_directions_param(direction)
            direction_df = load_direction_data()
            if not direction_df.empty and directions:
                # Filter direction data by specified directions and road_id
                filtered_directions = direction_df[
                    (direction_df['direction'].astype(str).str.upper().isin(directions)) &
                    (direction_df['road_id'] == road_id)
                ]

                # Get vehicle_ids and dates that match the direction filter
                if date:
                    filtered_directions = filtered_directions[filtered_directions['date'] == date]

                if not filtered_directions.empty:
                    # Create a set of (vehicle_id, date, seg_id) triplets for faster lookup
                    valid_triplets = set(
                        (int(v), str(d), int(s))
                        for v, d, s in zip(
                            filtered_directions['vehicle_id'],
                            filtered_directions['date'],
                            filtered_directions['seg_id'],
                        )
                    )

                    def dir_match(row):
                        try:
                            veh_id = int(row['vehicle_id'])
                        except Exception:
                            try:
                                veh_id = int(float(row['vehicle_id']))
                            except Exception:
                                veh_id = 0
                        try:
                            seg_val = int(row['seg_id'])
                        except Exception:
                            try:
                                seg_val = int(float(row['seg_id']))
                            except Exception:
                                seg_val = 0
                        return (veh_id, str(row['date']), seg_val) in valid_triplets

                    vehicle_data = vehicle_data[
                        vehicle_data.apply(dir_match, axis=1)
                    ]
                else:
                    # No vehicles match the direction filter
                    vehicle_data = vehicle_data.iloc[0:0]  # Empty dataframe with same structure
        # Movement-based filters removed
        
        # Convert to list of dictionaries format
        data_list = []
        for _, row in vehicle_data.iterrows():
            data_point = {
                'vehicle_id': int(row['vehicle_id']),
                'collectiontime': int(row['collectiontime']),
                'date': str(row['date']),
                'time_stamp': str(row['time_stamp']),
                'road_id': str(row['road_id']),
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'speed': float(row['speed']),
                'acceleratorpedal': float(row['acceleratorpedal']),
                'brakestatus': int(row['brakestatus'])
            }
            
            # seg_id if present (from split files)
            if 'seg_id' in row and pd.notna(row['seg_id']):
                try:
                    data_point['seg_id'] = int(row['seg_id'])
                except Exception:
                    data_point['seg_id'] = 0
                
            # Add optional new fields if they exist in the CSV
            if 'gearnum' in row and pd.notna(row['gearnum']) and str(row['gearnum']).strip():
                data_point['gearnum'] = str(row['gearnum'])
            else:
                data_point['gearnum'] = 'N/A'
                
            if 'havebrake' in row and pd.notna(row['havebrake']) and str(row['havebrake']).strip():
                data_point['havebrake'] = str(row['havebrake'])
            else:
                data_point['havebrake'] = 'N/A'
                
            if 'havedriver' in row and pd.notna(row['havedriver']) and str(row['havedriver']).strip():
                data_point['havedriver'] = str(row['havedriver'])
            else:
                data_point['havedriver'] = 'N/A'
            # Optional end_time (present in merged CSV for merged stationary points)
            if 'end_time' in row and pd.notna(row['end_time']) and str(row['end_time']).strip():
                data_point['end_time'] = str(row['end_time'])
            data_list.append(data_point)
        
        return data_list
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

class CustomHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Custom HTTP request handler for the vehicle data API
    """
    
    def _set_cors_headers(self):
        """Set CORS headers for cross-origin requests"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        
        response_json = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _parse_query_params(self):
        """Parse URL query parameters"""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Convert single-item lists to values; keep lists when multiple are provided
        params = {}
        for key, value_list in query_params.items():
            if not value_list:
                params[key] = None
            elif len(value_list) == 1:
                params[key] = value_list[0]
            else:
                params[key] = value_list
        
        return parsed_url.path, params
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests for CORS"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            path, params = self._parse_query_params()
            
            if path == '/':
                self._handle_index()
            elif path == '/health':
                self._handle_health_check()
            elif path == '/api/vehicle/data':
                self._handle_vehicle_data(params)
            elif path == '/api/vehicle/summary':
                self._handle_vehicle_summary(params)
            elif path == '/api/vehicle/dates':
                self._handle_available_dates(params)
            elif path == '/api/traffic/cycles':
                self._handle_traffic_cycles(params)
            elif path == '/api/traffic/status':
                self._handle_traffic_status(params)
            elif path == '/api/speed/analysis':
                self._handle_speed_analysis(params)
            elif path == '/api/speed/traffic-lights':
                self._handle_speed_traffic_lights(params)
            elif path == '/api/speed/time-range':
                self._handle_speed_time_range(params)
            elif path == '/api/topology/intersection':
                self._handle_intersection_topology(params)
            elif path == '/api/intersection/inference':
                self._handle_intersection_inference(params)
            elif path == '/api/intersection/centerlines':
                self._handle_intersection_centerlines()
            elif path == '/api/excluded/data':
                self._handle_excluded_data(params)
            elif path == '/api/direction/segments':
                self._handle_direction_segments(params)
            elif path == '/api/trajectory/segment':
                self._handle_trajectory_segment(params)
            elif path == '/api/raw/data':
                self._handle_raw_data(params)
            elif path == '/api/raw/dates':
                self._handle_raw_dates(params)
            else:
                self._send_json_response({
                    'status': 'error',
                    'message': 'Endpoint not found'
                }, 404)
                
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_index(self):
        """Handle root endpoint"""
        response = {
            'message': 'Geographic Data Visualization Backend Service',
            'version': '1.0.0',
            'endpoints': [
                '/api/vehicle/data - Get vehicle trajectory data',
                '/api/vehicle/summary - Get vehicle data summary',
                '/api/vehicle/dates - Get available dates',
                '/api/traffic/cycles - Get available cycles for traffic lights',
                '/api/traffic/status - Get traffic light status for specific cycle',
                '/api/speed/analysis - Get speed analysis data for specific vehicle or time range',
                '/api/speed/traffic-lights - Get traffic light data for speed analysis time range',
                '/api/topology/intersection - Get intersection topology GeoJSON',
                '/api/speed/time-range - Get available time range for specific date',
                '/api/raw/data - Get raw CSV trajectory data',
                '/api/raw/dates - Get available raw dates',
                '/health - Health check'
            ]
        }
        self._send_json_response(response)
    
    def _handle_health_check(self):
        """Handle health check endpoint"""
        response = {
            'status': 'healthy',
            'message': 'Server is running normally'
        }
        self._send_json_response(response)
    
    def _handle_vehicle_data(self, params):
        """Handle vehicle data endpoint"""
        try:
            # Get road ID from query parameters, default to A0003
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            
            # Get date filter from query parameters
            date = params.get('date')
            
            # Get direction/movement filters and source
            direction = params.get('direction')
            
            # Get data point limit, default to return all data
            limit = params.get('limit')
            if limit:
                limit = int(limit)
            
            # Check mode: time_range, batch, or single
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            vehicle_count_param = params.get('vehicle_count')
            
            if start_time and end_time:
                # Time range mode: load all vehicles in time range
                data = load_vehicle_data(
                    road_id=road_id,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    direction=direction,
                    source=source
                )
                vehicle_id = None  # Not applicable in time range mode
                vehicle_count = None  # Will be calculated from data
            elif vehicle_count_param:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count_param)
                data = load_vehicle_data(
                    road_id=road_id,
                    vehicle_count=vehicle_count,
                    date=date,
                    direction=direction,
                    source=source
                )
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                vehicle_count = None  # Not applicable in single mode
                data = load_vehicle_data(
                    vehicle_id=vehicle_id,
                    road_id=road_id,
                    date=date,
                    direction=direction,
                    source=source
                )
            
            if limit and limit > 0:
                data = data[:limit]
            
            # Build response based on mode
            if start_time and end_time:
                # Time range mode response
                unique_vehicles = list(set([point['vehicle_id'] for point in data]))
                response = {
                    'status': 'success',
                    'mode': 'time_range',
                    'start_time': start_time,
                    'end_time': end_time,
                    'vehicle_count': len(unique_vehicles),
                    'vehicle_ids': unique_vehicles,
                    'total_points': len(data),
                    'data': data
                }
            elif vehicle_count:
                # Batch mode response
                unique_vehicles = list(set([point['vehicle_id'] for point in data]))
                response = {
                    'status': 'success',
                    'mode': 'batch',
                    'vehicle_count': len(unique_vehicles),
                    'vehicle_ids': unique_vehicles,
                    'total_points': len(data),
                    'data': data
                }
            else:
                # Single mode response
                response = {
                    'status': 'success',
                    'mode': 'single',
                    'vehicle_id': vehicle_id,
                    'total_points': len(data),
                    'data': data
                }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_vehicle_summary(self, params):
        """Handle vehicle summary endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            
            # Get date filter from query parameters
            date = params.get('date')
            
            # Get direction filter from query parameters (single or multiple)
            direction = params.get('direction')
            
            # Check mode: time_range, batch, or single
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            vehicle_count_param = params.get('vehicle_count')
            
            if start_time and end_time:
                # Time range mode: load all vehicles in time range
                data = load_vehicle_data(
                    road_id=road_id,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    direction=direction,
                    source=source
                )
                vehicle_id = None  # Not applicable in time range mode
                vehicle_count = None  # Will be calculated from data
            elif vehicle_count_param:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count_param)
                data = load_vehicle_data(
                    road_id=road_id,
                    vehicle_count=vehicle_count,
                    date=date,
                    direction=direction,
                    source=source
                )
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                vehicle_count = None  # Not applicable in single mode
                data = load_vehicle_data(
                    vehicle_id=vehicle_id,
                    road_id=road_id,
                    date=date,
                    direction=direction,
                    source=source
                )
            
            if not data:
                self._send_json_response({
                    'status': 'error',
                    'message': 'No data found'
                }, 404)
                return
            
            # Calculate summary information
            longitudes = [point['longitude'] for point in data]
            latitudes = [point['latitude'] for point in data]
            speeds = [point['speed'] for point in data]
            
            # Build summary based on mode
            if start_time and end_time:
                # Time range mode summary
                unique_vehicles = list(set([point['vehicle_id'] for point in data]))
                summary = {
                    'status': 'success',
                    'mode': 'time_range',
                    'start_time': start_time,
                    'end_time': end_time,
                    'vehicle_count': len(unique_vehicles),
                    'vehicle_ids': unique_vehicles,
                    'total_points': len(data),
                    'time_range': {
                        'start': data[0]['time_stamp'],
                        'end': data[-1]['time_stamp']
                    },
                    'coordinate_bounds': {
                        'longitude_min': min(longitudes),
                        'longitude_max': max(longitudes),
                        'latitude_min': min(latitudes),
                        'latitude_max': max(latitudes)
                    },
                    'speed_stats': {
                        'min': min(speeds),
                        'max': max(speeds),
                        'avg': sum(speeds) / len(speeds)
                    }
                }
            elif vehicle_count:
                # Batch mode summary
                unique_vehicles = list(set([point['vehicle_id'] for point in data]))
                summary = {
                    'status': 'success',
                    'mode': 'batch',
                    'vehicle_count': len(unique_vehicles),
                    'vehicle_ids': unique_vehicles,
                    'total_points': len(data),
                    'time_range': {
                        'start': data[0]['time_stamp'],
                        'end': data[-1]['time_stamp']
                    },
                    'coordinate_bounds': {
                        'longitude_min': min(longitudes),
                        'longitude_max': max(longitudes),
                        'latitude_min': min(latitudes),
                        'latitude_max': max(latitudes)
                    },
                    'speed_stats': {
                        'min': min(speeds),
                        'max': max(speeds),
                        'avg': sum(speeds) / len(speeds)
                    }
                }
            else:
                # Single mode summary
                summary = {
                    'status': 'success',
                    'mode': 'single',
                    'vehicle_id': vehicle_id,
                    'total_points': len(data),
                    'time_range': {
                        'start': data[0]['time_stamp'],
                        'end': data[-1]['time_stamp']
                    },
                    'coordinate_bounds': {
                        'longitude_min': min(longitudes),
                        'longitude_max': max(longitudes),
                        'latitude_min': min(latitudes),
                        'latitude_max': max(latitudes)
                    },
                    'speed_stats': {
                        'min': min(speeds),
                        'max': max(speeds),
                        'avg': sum(speeds) / len(speeds)
                    }
                }
            
            self._send_json_response(summary)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_available_dates(self, params):
        """Handle available dates endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            
            # Build CSV file path based on road_id and selected source
            csv_file_path = _csv_path_for_source(road_id, source)
            
            if not os.path.exists(csv_file_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'CSV file for road {road_id} not found'
                }, 404)
                return
            
            # Read CSV file and get unique dates
            df = pd.read_csv(csv_file_path)
            unique_dates = sorted(df['date'].unique().tolist())
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'dates': unique_dates,
                'total_dates': len(unique_dates)
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_traffic_cycles(self, params):
        """Handle traffic cycles endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            
            # Check if traffic signal file exists
            if not os.path.exists(TRAFFIC_SIGNAL_FILE):
                self._send_json_response({
                    'status': 'error',
                    'message': f'Traffic signal file not found'
                }, 404)
                return
            
            # Read traffic signal data
            df = pd.read_csv(TRAFFIC_SIGNAL_FILE)
            
            # Get available cycles for the road
            road_data = df[df['road_id'] == road_id]
            if road_data.empty:
                cycles = []
            else:
                cycles = sorted(road_data['cycle_num'].unique().tolist())
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'cycles': cycles,
                'total_cycles': len(cycles)
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_traffic_status(self, params):
        """Handle traffic status endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            cycle_num = params.get('cycle_num')
            
            if not cycle_num:
                self._send_json_response({
                    'status': 'error',
                    'message': 'cycle_num parameter is required'
                }, 400)
                return
            
            try:
                cycle_num = int(cycle_num)
            except ValueError:
                self._send_json_response({
                    'status': 'error',
                    'message': 'cycle_num must be a valid integer'
                }, 400)
                return
            
            # Check if traffic signal file exists
            if not os.path.exists(TRAFFIC_SIGNAL_FILE):
                self._send_json_response({
                    'status': 'error',
                    'message': f'Traffic signal file not found'
                }, 404)
                return
            
            # Read traffic signal data
            df = pd.read_csv(TRAFFIC_SIGNAL_FILE)
            
            # Filter by road_id and cycle_num
            cycle_data = df[(df['road_id'] == road_id) & (df['cycle_num'] == cycle_num)]
            
            if cycle_data.empty:
                self._send_json_response({
                    'status': 'error',
                    'message': f'No data found for road {road_id} cycle {cycle_num}'
                }, 404)
                return
            
            # Convert to list of dictionaries
            traffic_lights = []
            for _, row in cycle_data.iterrows():
                traffic_lights.append({
                    'road_id': row['road_id'],
                    'phase_id': row['phase_id'],
                    'cycle_num': int(row['cycle_num']),
                    'start_time': row['start_time'],
                    'end_time': row['end_time']
                })
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'cycle_num': cycle_num,
                'phases': traffic_lights,
                'total_phases': len(traffic_lights)
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_speed_analysis(self, params):
        """Handle speed analysis endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            vehicle_id = params.get('vehicle_id')
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            date = params.get('date')
            direction = params.get('direction')  # Single or multiple
            seg_id_param = params.get('seg_id')
            
            if not date:
                self._send_json_response({
                    'status': 'error',
                    'message': 'date parameter is required'
                }, 400)
                return
            
            # Check if this is time range mode or single vehicle mode
            if start_time and end_time:
                # Time range mode: analyze all vehicles in time range
                mode = 'time_range'
            else:
                # Single mode: analyze specific vehicle
                if not vehicle_id:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'vehicle_id parameter is required for single mode'
                    }, 400)
                    return
                
                try:
                    vehicle_id = int(vehicle_id)
                except ValueError:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'vehicle_id must be a valid integer'
                    }, 400)
                    return
                    
                mode = 'single'
            
            # Build CSV file path based on road_id and selected source
            csv_file_path = _csv_path_for_source(road_id, source)
            
            if not os.path.exists(csv_file_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'CSV file for road {road_id} not found'
                }, 404)
                return
            
            # Read CSV file and filter data
            df = pd.read_csv(csv_file_path)
            
            # Apply date filter first
            df_filtered = df[df['date'] == date]
            
            if df_filtered.empty:
                self._send_json_response({
                    'status': 'error',
                    'message': f'No data found for date {date}'
                }, 404)
                return
            
            # Apply filter based on mode
            if mode == 'time_range':
                # Time range mode: filter by time range
                # Convert time strings to time objects for comparison
                try:
                    import datetime
                    start_time_obj = datetime.datetime.strptime(start_time, '%H:%M').time()
                    end_time_obj = datetime.datetime.strptime(end_time, '%H:%M').time()
                    
                    # Filter data by time range
                    def time_in_range(time_stamp):
                        try:
                            time_obj = datetime.datetime.strptime(time_stamp, '%H:%M:%S').time()
                            return start_time_obj <= time_obj <= end_time_obj
                        except:
                            return False
                    
                    vehicle_data = df_filtered[df_filtered['time_stamp'].apply(time_in_range)]
                    
                except ValueError:
                    self._send_json_response({
                        'status': 'error',
                        'message': 'Invalid time format. Use HH:MM format for start_time and end_time'
                    }, 400)
                    return
            else:
                # Single mode: filter by vehicle ID
                vehicle_data = df_filtered[df_filtered['vehicle_id'] == vehicle_id]
            
            # Apply direction filter if specified (single or multiple)
            if direction:
                directions = parse_directions_param(direction)
                direction_df = load_direction_data()
                if not direction_df.empty and directions:
                    # Filter direction data by the specified directions and road_id
                    filtered_directions = direction_df[
                        (direction_df['direction'].astype(str).str.upper().isin(directions)) & 
                        (direction_df['road_id'] == road_id)
                    ]
                    
                    # Get vehicle_ids and dates that match the direction filter
                    if date:
                        # If date is specified, filter by date as well
                        filtered_directions = filtered_directions[filtered_directions['date'] == date]
                    
                    if not filtered_directions.empty:
                        # Create a set of (vehicle_id, date, seg_id) triplets for faster lookup
                        valid_triplets = set(
                            (int(v), str(d), int(s))
                            for v, d, s in zip(
                                filtered_directions['vehicle_id'],
                                filtered_directions['date'],
                                filtered_directions['seg_id'],
                            )
                        )
                        
                        # Filter vehicle_data to only include rows with matching (vehicle_id, date, seg_id)
                        vehicle_data = vehicle_data[
                            vehicle_data.apply(
                                lambda row: (int(row['vehicle_id']), str(row['date']), int(row['seg_id'])) in valid_triplets,
                                axis=1,
                            )
                        ]
                    else:
                        # No vehicles match the direction filter
                        vehicle_data = vehicle_data.iloc[0:0]  # Empty dataframe with same structure
            
            if vehicle_data.empty:
                if mode == 'time_range':
                    if direction:
                        dirs_text = ','.join(parse_directions_param(direction))
                        error_msg = f'No data found in time range {start_time}-{end_time} on date {date} with direction {dirs_text}'
                    else:
                        error_msg = f'No data found in time range {start_time}-{end_time} on date {date}'
                else:
                    if direction:
                        dirs_text = ','.join(parse_directions_param(direction))
                        error_msg = f'No data found for vehicle {vehicle_id} on date {date} with direction {dirs_text}'
                    else:
                        error_msg = f'No data found for vehicle {vehicle_id} on date {date}'
                    
                self._send_json_response({
                    'status': 'error',
                    'message': error_msg
                }, 404)
                return
            
            # Sort by time for proper time series
            vehicle_data = vehicle_data.sort_values('collectiontime')
            
            # Optional seg_id filtering
            if seg_id_param is not None:
                try:
                    seg_id_int = int(seg_id_param)
                    if 'seg_id' in vehicle_data.columns:
                        vehicle_data = vehicle_data[vehicle_data['seg_id'] == seg_id_int]
                except ValueError:
                    pass

            # Convert to list of dictionaries with all metrics for charting
            speed_data = []
            for _, row in vehicle_data.iterrows():
                point = {
                    'vehicle_id': int(row['vehicle_id']),
                    'collectiontime': int(row['collectiontime']),
                    'date': str(row['date']),
                    'time_stamp': str(row['time_stamp']),
                    'road_id': str(row['road_id']),
                    'speed': float(row['speed']) if 'speed' in row and pd.notna(row['speed']) else None
                }
                # seg_id if present
                if 'seg_id' in row and pd.notna(row['seg_id']):
                    try:
                        point['seg_id'] = int(row['seg_id'])
                    except Exception:
                        point['seg_id'] = None
                # include other optional metrics if present
                if 'acceleratorpedal' in row and pd.notna(row['acceleratorpedal']):
                    try:
                        point['acceleratorpedal'] = float(row['acceleratorpedal'])
                    except Exception:
                        point['acceleratorpedal'] = None
                if 'brakestatus' in row and pd.notna(row['brakestatus']):
                    try:
                        point['brakestatus'] = int(row['brakestatus'])
                    except Exception:
                        point['brakestatus'] = None
                if 'gearnum' in row and pd.notna(row['gearnum']):
                    point['gearnum'] = str(row['gearnum'])
                if 'havebrake' in row and pd.notna(row['havebrake']):
                    point['havebrake'] = str(row['havebrake'])
                if 'havedriver' in row and pd.notna(row['havedriver']):
                    point['havedriver'] = str(row['havedriver'])
                speed_data.append(point)
            
            # Build response based on mode
            if mode == 'time_range':
                unique_vehicles = list(set([point['vehicle_id'] for point in speed_data]))
                response = {
                    'status': 'success',
                    'mode': 'time_range',
                    'road_id': road_id,
                    'date': date,
                    'start_time': start_time,
                    'end_time': end_time,
                    'vehicle_count': len(unique_vehicles),
                    'vehicle_ids': sorted(unique_vehicles),
                    'total_points': len(speed_data),
                    'data': speed_data
                }
            else:
                response = {
                    'status': 'success',
                    'mode': 'single',
                    'road_id': road_id,
                    'vehicle_id': vehicle_id,
                    'date': date,
                    'total_points': len(speed_data),
                    'data': speed_data
                }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_speed_traffic_lights(self, params):
        """Handle traffic lights data for speed analysis endpoint"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            
            if not start_time or not end_time:
                self._send_json_response({
                    'status': 'error',
                    'message': 'start_time and end_time parameters are required'
                }, 400)
                return
            
            # Check if traffic signal file exists
            if not os.path.exists(TRAFFIC_SIGNAL_FILE):
                self._send_json_response({
                    'status': 'error',
                    'message': 'Traffic signal file not found'
                }, 404)
                return
            
            # Read traffic signal data
            df = pd.read_csv(TRAFFIC_SIGNAL_FILE)
            
            # Filter by road_id
            road_data = df[df['road_id'] == road_id]
            
            if road_data.empty:
                self._send_json_response({
                    'status': 'success',
                    'road_id': road_id,
                    'traffic_lights': [],
                    'total_lights': 0
                })
                return
            
            # Convert time strings to datetime for comparison
            road_data = road_data.copy()
            road_data['start_datetime'] = pd.to_datetime(road_data['start_time'])
            road_data['end_datetime'] = pd.to_datetime(road_data['end_time'])
            
            start_datetime = pd.to_datetime(start_time)
            end_datetime = pd.to_datetime(end_time)
            
            # Filter by time range - find traffic lights that overlap with the speed analysis time range
            overlapping_lights = road_data[
                (road_data['start_datetime'] <= end_datetime) & 
                (road_data['end_datetime'] >= start_datetime)
            ]
            
            # Convert to list of dictionaries
            traffic_lights = []
            for _, row in overlapping_lights.iterrows():
                traffic_lights.append({
                    'road_id': row['road_id'],
                    'phase_id': row['phase_id'],
                    'cycle_num': int(row['cycle_num']),
                    'start_time': row['start_time'],
                    'end_time': row['end_time']
                })
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'start_time': start_time,
                'end_time': end_time,
                'traffic_lights': traffic_lights,
                'total_lights': len(traffic_lights)
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _handle_speed_time_range(self, params):
        """Handle speed time range endpoint to get available time range for a date"""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            date = params.get('date')
            
            if not date:
                self._send_json_response({
                    'status': 'error',
                    'message': 'date parameter is required'
                }, 400)
                return
            
            # Build CSV file path based on road_id and selected source (default prefer merged)
            csv_file_path = _csv_path_for_source(road_id, source)
            
            if not os.path.exists(csv_file_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'CSV file for road {road_id} not found'
                }, 404)
                return
            
            # Read CSV file and filter by date
            df = pd.read_csv(csv_file_path)
            date_data = df[df['date'] == date]
            
            if date_data.empty:
                self._send_json_response({
                    'status': 'error',
                    'message': f'No data found for date {date}'
                }, 404)
                return
            
            # Get time range from the data
            time_stamps = date_data['time_stamp'].tolist()
            min_time = min(time_stamps)
            max_time = max(time_stamps)
            
            # Get unique vehicle count
            unique_vehicles = date_data['vehicle_id'].nunique()
            total_records = len(date_data)
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'date': date,
                'min_time': min_time,
                'max_time': max_time,
                'total_vehicles': unique_vehicles,
                'total_records': total_records,
                'time_range_duration': self._calculate_time_duration(min_time, max_time)
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def _calculate_time_duration(self, start_time, end_time):
        """Calculate duration between two time strings"""
        try:
            import datetime
            start_obj = datetime.datetime.strptime(start_time, '%H:%M:%S')
            end_obj = datetime.datetime.strptime(end_time, '%H:%M:%S')
            
            # Handle case where end time is next day
            if end_obj < start_obj:
                end_obj += datetime.timedelta(days=1)
            
            duration = end_obj - start_obj
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except:
            return "Unknown"
    
    def _handle_intersection_topology(self, params):
        """Handle intersection topology endpoint to load GeoJSON road network"""
        try:
            road_id = params.get('road_id', 'A0003')
            
            # Path to the output directory containing the geojson file
            topology_file = os.path.join(CSV_BASE_PATH, 'output', 'intersection_topology.geojson')
            # Fallback to repo root output if not found under data
            if not os.path.exists(topology_file):
                alt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'intersection_topology.geojson'))
                if os.path.exists(alt):
                    topology_file = alt
            
            print(f"Loading intersection topology from: {topology_file}")
            
            # Check if file exists
            if not os.path.exists(topology_file):
                self._send_json_response({
                    'status': 'error',
                    'message': f'Topology file not found: {topology_file}',
                    'hint': 'Please run infer_intersection.py first to generate the topology file'
                }, 404)
                return
            
            # Read the GeoJSON file
            with open(topology_file, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # Also load the summary file if available
            summary_file = os.path.join(CSV_BASE_PATH, 'output', 'summary.json')
            if not os.path.exists(summary_file):
                alt_sum = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'summary.json'))
                if os.path.exists(alt_sum):
                    summary_file = alt_sum
            summary_data = None
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            
            response = {
                'status': 'success',
                'road_id': road_id,
                'topology': geojson_data,
                'summary': summary_data,
                'file_path': topology_file
            }
            
            print(f"Successfully loaded intersection topology with {len(geojson_data.get('features', []))} features")
            
            self._send_json_response(response)
            
        except json.JSONDecodeError as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Invalid JSON in topology file: {str(e)}'
            }, 500)
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Error loading topology: {str(e)}'
            }, 500)

    def _handle_intersection_inference(self, params):
        """Serve intersection inference JSON (center, axes, stoplines) per road_id.
        Expects files like A0003_intersection.json, A0008_intersection.json under CSV_BASE_PATH.
        """
        try:
            road_id = params.get('road_id', 'A0003')
            json_path = os.path.join(CSV_BASE_PATH, f'{road_id}_intersection.json')
            # Fallback to repo root data if not found under data
            if not os.path.exists(json_path):
                alt_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', f'{road_id}_intersection.json'))
                if os.path.exists(alt_json):
                    json_path = alt_json

            if not os.path.exists(json_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'Inference file not found for road {road_id}: {json_path}',
                    'hint': 'Run main.py to generate data/{road}_intersection.json via Step 5'
                }, 404)
                return

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._send_json_response({
                'status': 'success',
                'road_id': road_id,
                'inference': data,
                'file_path': json_path
            })
        except json.JSONDecodeError as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Invalid JSON in inference file: {str(e)}'
            }, 500)
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Error loading inference: {str(e)}'
            }, 500)

    def _handle_intersection_centerlines(self):
        """Serve all intersections' centerlines from data/intersection.json.
        Expects the file to contain a mapping of road_id -> { center, lower_lane, upper_lane } with
        each value being an array of [lon, lat] pairs.
        """
        try:
            # Prefer data/intersection.json under repository root
            inter_path = os.path.join(CSV_BASE_PATH, 'intersection.json')
            # Fallback to repo root absolute if not found under data
            if not os.path.exists(inter_path):
                alt_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'intersection.json'))
                if os.path.exists(alt_json):
                    inter_path = alt_json

            if not os.path.exists(inter_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'Centerlines file not found: {inter_path}'
                }, 404)
                return

            with open(inter_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                self._send_json_response({
                    'status': 'error',
                    'message': 'Invalid intersection.json format (expected object mapping road_id to lines)'
                }, 400)
                return

            roads = list(payload.keys())
            self._send_json_response({
                'status': 'success',
                'centerlines': payload,
                'roads': roads,
                'total': len(roads),
                'file_path': inter_path
            })
        except json.JSONDecodeError as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Invalid JSON in centerlines file: {str(e)}'
            }, 500)
        except Exception as e:
            self._send_json_response({
                'status': 'error',
                'message': f'Error loading centerlines: {str(e)}'
            }, 500)

    def _handle_raw_dates(self, params):
        """Return available dates from the raw CSV (<road_id>.csv)."""
        try:
            road_id = params.get('road_id', 'A0003')
            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')

            if not os.path.exists(csv_file_path):
                self._send_json_response({
                    'status': 'error',
                    'message': f'CSV file for road {road_id} not found'
                }, 404)
                return

            df = pd.read_csv(csv_file_path)
            if 'road_id' in df.columns:
                df = df[df['road_id'] == road_id]
            unique_dates = sorted(df['date'].unique().tolist()) if 'date' in df.columns else []

            self._send_json_response({
                'status': 'success',
                'road_id': road_id,
                'dates': unique_dates,
                'total_dates': len(unique_dates)
            })
        except Exception as e:
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)

    def _handle_raw_data(self, params):
        """Return raw trajectory data from <road_id>.csv filtered by date and optional vehicle_id."""
        try:
            road_id = params.get('road_id', 'A0003')
            date = params.get('date')
            vehicle_id_param = params.get('vehicle_id')

            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
            if not os.path.exists(csv_file_path):
                self._send_json_response({'status': 'error', 'message': f'CSV file for road {road_id} not found'}, 404)
                return

            df = pd.read_csv(csv_file_path)

            # Filter to requested road if column exists
            if 'road_id' in df.columns:
                df = df[df['road_id'] == road_id]

            # Filter by date if provided
            if date and 'date' in df.columns:
                df = df[df['date'] == date]

            # Filter by vehicle if provided
            if vehicle_id_param and 'vehicle_id' in df.columns:
                try:
                    vid = int(vehicle_id_param)
                    df = df[df['vehicle_id'] == vid]
                except ValueError:
                    pass

            if df.empty:
                self._send_json_response({
                    'status': 'success',
                    'road_id': road_id,
                    'date': date,
                    'vehicle_id': vehicle_id_param,
                    'total_points': 0,
                    'data': []
                })
                return

            # Order by typical keys when present
            sort_cols = [c for c in ['vehicle_id', 'date', 'seg_id', 'collectiontime'] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)

            data_list = []
            for _, row in df.iterrows():
                point = {
                    'vehicle_id': int(row['vehicle_id']) if 'vehicle_id' in row and pd.notna(row['vehicle_id']) else 0,
                    'collectiontime': int(row['collectiontime']) if 'collectiontime' in row and pd.notna(row['collectiontime']) else 0,
                    'date': str(row['date']) if 'date' in row and pd.notna(row['date']) else '',
                    'time_stamp': str(row['time_stamp']) if 'time_stamp' in row and pd.notna(row['time_stamp']) else '',
                    'road_id': str(row['road_id']) if 'road_id' in row and pd.notna(row['road_id']) else road_id,
                    'longitude': float(row['longitude']) if 'longitude' in row and pd.notna(row['longitude']) else 0.0,
                    'latitude': float(row['latitude']) if 'latitude' in row and pd.notna(row['latitude']) else 0.0,
                    'speed': float(row['speed']) if 'speed' in row and pd.notna(row['speed']) else 0.0,
                    'acceleratorpedal': float(row['acceleratorpedal']) if 'acceleratorpedal' in row and pd.notna(row['acceleratorpedal']) else 0.0,
                    'brakestatus': int(row['brakestatus']) if 'brakestatus' in row and pd.notna(row['brakestatus']) else 0,
                }

                if 'seg_id' in row and pd.notna(row['seg_id']):
                    try:
                        point['seg_id'] = int(row['seg_id'])
                    except Exception:
                        point['seg_id'] = 0

                if 'gearnum' in row and pd.notna(row['gearnum']) and str(row['gearnum']).strip():
                    point['gearnum'] = str(row['gearnum'])
                else:
                    point['gearnum'] = 'N/A'

                if 'havebrake' in row and pd.notna(row['havebrake']) and str(row['havebrake']).strip():
                    point['havebrake'] = str(row['havebrake'])
                else:
                    point['havebrake'] = 'N/A'

                if 'havedriver' in row and pd.notna(row['havedriver']) and str(row['havedriver']).strip():
                    point['havedriver'] = str(row['havedriver'])
                else:
                    point['havedriver'] = 'N/A'

                data_list.append(point)

            self._send_json_response({
                'status': 'success',
                'road_id': road_id,
                'date': date,
                'vehicle_id': vehicle_id_param,
                'total_points': len(data_list),
                'data': data_list
            })
        except Exception as e:
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)

    def _handle_excluded_data(self, params):
        """Return excluded trajectory points filtered by road_id and optional date/vehicle_id."""
        try:
            road_id = params.get('road_id', 'A0003')
            date = params.get('date')
            vehicle_id_param = params.get('vehicle_id')

            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}_excluded.csv')
            if not os.path.exists(csv_file_path):
                self._send_json_response({
                    'status': 'success',
                    'road_id': road_id,
                    'date': date,
                    'total_points': 0,
                    'data': []
                })
                return

            df = pd.read_csv(csv_file_path)
            if 'road_id' in df.columns:
                df = df[df['road_id'] == road_id]
            if date:
                df = df[df['date'] == date]
            if vehicle_id_param:
                try:
                    vid = int(vehicle_id_param)
                    df = df[df['vehicle_id'] == vid]
                except ValueError:
                    pass

            if df.empty:
                self._send_json_response({
                    'status': 'success',
                    'road_id': road_id,
                    'date': date,
                    'total_points': 0,
                    'data': []
                })
                return

            # Stable ordering
            sort_cols = [c for c in ['vehicle_id', 'date', 'seg_id', 'collectiontime'] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)

            data_list = []
            for _, row in df.iterrows():
                data_point = {
                    'vehicle_id': int(row['vehicle_id']),
                    'collectiontime': int(row['collectiontime']),
                    'date': str(row['date']),
                    'time_stamp': str(row['time_stamp']),
                    'road_id': str(row['road_id']),
                    'longitude': float(row['longitude']),
                    'latitude': float(row['latitude']),
                    'speed': float(row['speed']) if 'speed' in row and pd.notna(row['speed']) else 0.0,
                    'brakestatus': int(row['brakestatus']) if 'brakestatus' in row and pd.notna(row['brakestatus']) else 0
                }

                if 'seg_id' in row and pd.notna(row['seg_id']):
                    try:
                        data_point['seg_id'] = int(row['seg_id'])
                    except Exception:
                        data_point['seg_id'] = 0

                if 'acceleratorpedal' in row and pd.notna(row['acceleratorpedal']):
                    try:
                        data_point['acceleratorpedal'] = float(row['acceleratorpedal'])
                    except Exception:
                        data_point['acceleratorpedal'] = None
                else:
                    data_point['acceleratorpedal'] = None

                if 'gearnum' in row and pd.notna(row['gearnum']) and str(row['gearnum']).strip():
                    data_point['gearnum'] = str(row['gearnum'])
                else:
                    data_point['gearnum'] = 'N/A'

                if 'havebrake' in row and pd.notna(row['havebrake']) and str(row['havebrake']).strip():
                    data_point['havebrake'] = str(row['havebrake'])
                else:
                    data_point['havebrake'] = 'N/A'

                if 'havedriver' in row and pd.notna(row['havedriver']) and str(row['havedriver']).strip():
                    data_point['havedriver'] = str(row['havedriver'])
                else:
                    data_point['havedriver'] = 'N/A'

                data_list.append(data_point)

            # total segments
            total_segments = 0
            if {'vehicle_id', 'date', 'seg_id'}.issubset(df.columns):
                total_segments = df.drop_duplicates(['vehicle_id', 'date', 'seg_id']).shape[0]

            self._send_json_response({
                'status': 'success',
                'road_id': road_id,
                'date': date,
                'vehicle_id': int(vehicle_id_param) if vehicle_id_param and str(vehicle_id_param).isdigit() else None,
                'total_points': len(data_list),
                'total_segments': int(total_segments),
                'data': data_list
            })
        except Exception as e:
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)

    def _handle_direction_segments(self, params):
        """List segments filtered by road_id and direction (and optional vehicle_id/date/seg_id)."""
        try:
            road_id = params.get('road_id', 'A0003')
            direction = params.get('direction')
            vehicle_id = params.get('vehicle_id')
            date = params.get('date')
            seg_id = params.get('seg_id')

            df = load_direction_data()
            if df is None or df.empty:
                self._send_json_response({'status': 'success', 'segments': [], 'total': 0})
                return

            filt = (df['road_id'] == road_id)
            if direction:
                filt &= (df['direction'] == direction)
            if vehicle_id:
                try:
                    vid = int(vehicle_id)
                    filt &= (df['vehicle_id'] == vid)
                except ValueError:
                    pass
            if date:
                filt &= (df['date'] == date)
            if seg_id:
                try:
                    sid = int(seg_id)
                    filt &= (df['seg_id'] == sid)
                except ValueError:
                    pass

            df_out = df[filt].copy()
            df_out = df_out.sort_values(['vehicle_id', 'date', 'seg_id'])
            segments = [
                {
                    'vehicle_id': int(r['vehicle_id']),
                    'date': str(r['date']),
                    'seg_id': int(r['seg_id']),
                    'road_id': str(r['road_id']),
                    'direction': str(r['direction'])
                }
                for _, r in df_out.iterrows()
            ]

            self._send_json_response({'status': 'success', 'road_id': road_id, 'direction': direction, 'total': len(segments), 'segments': segments})
        except Exception as e:
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)

    def _handle_trajectory_segment(self, params):
        """Return time series for a specific (vehicle_id, date, seg_id) on a road."""
        try:
            road_id = params.get('road_id', 'A0003')
            source = params.get('source')
            vehicle_id = params.get('vehicle_id')
            date = params.get('date')
            seg_id = params.get('seg_id')

            if not (vehicle_id and date and seg_id):
                self._send_json_response({'status': 'error', 'message': 'vehicle_id, date, seg_id are required'}, 400)
                return

            try:
                vehicle_id_i = int(vehicle_id)
                seg_id_i = int(seg_id)
            except ValueError:
                self._send_json_response({'status': 'error', 'message': 'vehicle_id and seg_id must be integers'}, 400)
                return

            csv_file_path = _csv_path_for_source(road_id, source)
            if not os.path.exists(csv_file_path):
                self._send_json_response({'status': 'error', 'message': f'CSV file for road {road_id} not found'}, 404)
                return

            df = pd.read_csv(csv_file_path)
            df_seg = df[(df['vehicle_id'] == vehicle_id_i) & (df['date'] == date) & (df['seg_id'] == seg_id_i)].copy()

            if df_seg.empty:
                self._send_json_response({'status': 'error', 'message': 'No data for specified segment'}, 404)
                return

            df_seg = df_seg.sort_values('collectiontime')

            rows = []
            for _, r in df_seg.iterrows():
                row = {
                    'vehicle_id': int(r['vehicle_id']),
                    'collectiontime': int(r['collectiontime']),
                    'date': str(r['date']),
                    'time_stamp': str(r['time_stamp']),
                    'road_id': str(r['road_id']),
                    'seg_id': int(r['seg_id']),
                    'speed': float(r['speed']) if pd.notna(r['speed']) else None,
                    'acceleratorpedal': float(r['acceleratorpedal']) if 'acceleratorpedal' in r and pd.notna(r['acceleratorpedal']) else None,
                    'brakestatus': int(r['brakestatus']) if 'brakestatus' in r and pd.notna(r['brakestatus']) else None,
                    'gearnum': (str(r['gearnum']) if 'gearnum' in r and pd.notna(r['gearnum']) else 'N/A'),
                    'havebrake': (str(r['havebrake']) if 'havebrake' in r and pd.notna(r['havebrake']) else 'N/A'),
                    'havedriver': (str(r['havedriver']) if 'havedriver' in r and pd.notna(r['havedriver']) else 'N/A'),
                }
                if 'end_time' in r and pd.notna(r['end_time']) and str(r['end_time']).strip():
                    row['end_time'] = str(r['end_time'])
                rows.append(row)

            self._send_json_response({'status': 'success', 'road_id': road_id, 'vehicle_id': vehicle_id_i, 'date': date, 'seg_id': seg_id_i, 'total_points': len(rows), 'data': rows})
        except Exception as e:
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)

    
    
    def log_message(self, format, *args):
        """Override log message to show custom format"""
        print(f"[{self.address_string()}] {format % args}")

def run_server(host='127.0.0.1', port=5555):
    """Run the HTTP server"""
    print("Starting Geographic Data Visualization Backend Service...")
    print(f"CSV base path: {os.path.abspath(CSV_BASE_PATH)}")
    
    # Check if split CSV files exist
    road_ids = ['A0003', 'A0008']
    for road_id in road_ids:
        csv_file = os.path.join(CSV_BASE_PATH, f'{road_id}_split.csv')
        if os.path.exists(csv_file):
            print(f"✓ Found split CSV file: {road_id}_split.csv")
        else:
            print(f"⚠ Warning: split CSV file {road_id}_split.csv does not exist in {os.path.abspath(CSV_BASE_PATH)}")
    
    # Create and start server
    # Allow address reuse to prevent "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), CustomHTTPRequestHandler) as httpd:
        print(f"Server running at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped by user")

if __name__ == '__main__':
    run_server()