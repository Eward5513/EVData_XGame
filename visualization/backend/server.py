#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs

# CSV file base path
CSV_BASE_PATH = '../../'
TRAFFIC_SIGNAL_FILE = '../../traffic_signal.csv'
DIRECTION_FILE = '../../direction.csv'

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

def load_vehicle_data(vehicle_id=None, road_id='A0003', vehicle_count=None, date=None, start_time=None, end_time=None, direction=None):
    """
    Load data for specified vehicle ID(s) and road ID using pandas
    
    Args:
        vehicle_id: Single vehicle ID to load (for single mode)
        road_id: Road intersection ID 
        vehicle_count: Number of vehicles to load from the beginning (for batch mode)
        date: Date filter in YYYY-MM-DD format (optional)
        start_time: Start time filter in HH:MM format (optional)
        end_time: End time filter in HH:MM format (optional)
        direction: Direction filter (A1, A2, A3, A4, C) (optional)
    """
    try:
        # Build CSV file path based on road_id
        csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
        
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
        
        # Apply direction filter if specified
        if direction:
            direction_df = load_direction_data()
            if not direction_df.empty:
                # Filter direction data by the specified direction and road_id
                filtered_directions = direction_df[
                    (direction_df['direction'] == direction) & 
                    (direction_df['road_id'] == road_id)
                ]
                
                # Get vehicle_ids and dates that match the direction filter
                if date:
                    # If date is specified, filter by date as well
                    filtered_directions = filtered_directions[filtered_directions['date'] == date]
                
                if not filtered_directions.empty:
                    # Create a set of (vehicle_id, date) tuples for faster lookup
                    valid_combinations = set(zip(filtered_directions['vehicle_id'], filtered_directions['date']))
                    
                    # Filter vehicle_data to only include vehicles with matching direction
                    vehicle_data = vehicle_data[
                        vehicle_data.apply(lambda row: (row['vehicle_id'], row['date']) in valid_combinations, axis=1)
                    ]
                else:
                    # No vehicles match the direction filter
                    vehicle_data = vehicle_data.iloc[0:0]  # Empty dataframe with same structure
        
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
        
        # Convert single-item lists to values
        params = {}
        for key, value_list in query_params.items():
            params[key] = value_list[0] if value_list else None
        
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
                '/api/speed/time-range - Get available time range for specific date',
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
            
            # Get date filter from query parameters
            date = params.get('date')
            
            # Get direction filter from query parameters
            direction = params.get('direction')
            
            # Get data point limit, default to return all data
            limit = params.get('limit')
            if limit:
                limit = int(limit)
            
            # Check mode: time_range, batch, or single
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            vehicle_count = params.get('vehicle_count')
            
            if start_time and end_time:
                # Time range mode: load all vehicles in time range
                data = load_vehicle_data(road_id=road_id, date=date, start_time=start_time, end_time=end_time, direction=direction)
                vehicle_id = None  # Not applicable in time range mode
                vehicle_count = None  # Will be calculated from data
            elif vehicle_count:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count)
                data = load_vehicle_data(road_id=road_id, vehicle_count=vehicle_count, date=date, direction=direction)
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                data = load_vehicle_data(vehicle_id=vehicle_id, road_id=road_id, date=date, direction=direction)
            
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
            
            # Get date filter from query parameters
            date = params.get('date')
            
            # Get direction filter from query parameters
            direction = params.get('direction')
            
            # Check mode: time_range, batch, or single
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            vehicle_count = params.get('vehicle_count')
            
            if start_time and end_time:
                # Time range mode: load all vehicles in time range
                data = load_vehicle_data(road_id=road_id, date=date, start_time=start_time, end_time=end_time, direction=direction)
                vehicle_id = None  # Not applicable in time range mode
                vehicle_count = None  # Will be calculated from data
            elif vehicle_count:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count)
                data = load_vehicle_data(road_id=road_id, vehicle_count=vehicle_count, date=date, direction=direction)
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                data = load_vehicle_data(vehicle_id=vehicle_id, road_id=road_id, date=date, direction=direction)
            
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
            if vehicle_count:
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
            
            # Build CSV file path based on road_id
            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
            
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
            vehicle_id = params.get('vehicle_id')
            start_time = params.get('start_time')
            end_time = params.get('end_time')
            date = params.get('date')
            direction = params.get('direction')  # Get direction filter
            
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
            
            # Build CSV file path based on road_id
            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
            
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
            
            # Apply direction filter if specified
            if direction:
                direction_df = load_direction_data()
                if not direction_df.empty:
                    # Filter direction data by the specified direction and road_id
                    filtered_directions = direction_df[
                        (direction_df['direction'] == direction) & 
                        (direction_df['road_id'] == road_id)
                    ]
                    
                    # Get vehicle_ids and dates that match the direction filter
                    if date:
                        # If date is specified, filter by date as well
                        filtered_directions = filtered_directions[filtered_directions['date'] == date]
                    
                    if not filtered_directions.empty:
                        # Create a set of (vehicle_id, date) tuples for faster lookup
                        valid_combinations = set(zip(filtered_directions['vehicle_id'], filtered_directions['date']))
                        
                        # Filter vehicle_data to only include vehicles with matching direction
                        vehicle_data = vehicle_data[
                            vehicle_data.apply(lambda row: (row['vehicle_id'], row['date']) in valid_combinations, axis=1)
                        ]
                    else:
                        # No vehicles match the direction filter
                        vehicle_data = vehicle_data.iloc[0:0]  # Empty dataframe with same structure
            
            if vehicle_data.empty:
                if mode == 'time_range':
                    if direction:
                        error_msg = f'No data found in time range {start_time}-{end_time} on date {date} with direction {direction}'
                    else:
                        error_msg = f'No data found in time range {start_time}-{end_time} on date {date}'
                else:
                    if direction:
                        error_msg = f'No data found for vehicle {vehicle_id} on date {date} with direction {direction}'
                    else:
                        error_msg = f'No data found for vehicle {vehicle_id} on date {date}'
                    
                self._send_json_response({
                    'status': 'error',
                    'message': error_msg
                }, 404)
                return
            
            # Sort by time for proper time series
            vehicle_data = vehicle_data.sort_values('collectiontime')
            
            # Convert to list of dictionaries for speed analysis
            speed_data = []
            for _, row in vehicle_data.iterrows():
                speed_data.append({
                    'vehicle_id': int(row['vehicle_id']),
                    'collectiontime': int(row['collectiontime']),
                    'date': str(row['date']),
                    'time_stamp': str(row['time_stamp']),
                    'road_id': str(row['road_id']),
                    'speed': float(row['speed'])
                })
            
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
            date = params.get('date')
            
            if not date:
                self._send_json_response({
                    'status': 'error',
                    'message': 'date parameter is required'
                }, 400)
                return
            
            # Build CSV file path based on road_id
            csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
            
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
    
    def log_message(self, format, *args):
        """Override log message to show custom format"""
        print(f"[{self.address_string()}] {format % args}")

def run_server(host='127.0.0.1', port=5000):
    """Run the HTTP server"""
    print("Starting Geographic Data Visualization Backend Service...")
    print(f"CSV base path: {os.path.abspath(CSV_BASE_PATH)}")
    
    # Check if CSV files exist
    road_ids = ['A0003', 'A0008']
    for road_id in road_ids:
        csv_file = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
        if os.path.exists(csv_file):
            print(f"✓ Found CSV file: {road_id}.csv")
        else:
            print(f"⚠ Warning: CSV file {road_id}.csv does not exist")
    
    # Create and start server
    with socketserver.TCPServer((host, port), CustomHTTPRequestHandler) as httpd:
        print(f"Server running at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped by user")

if __name__ == '__main__':
    run_server()