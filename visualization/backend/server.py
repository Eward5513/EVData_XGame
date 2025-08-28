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

def load_vehicle_data(vehicle_id=None, road_id='A0003', vehicle_count=None):
    """
    Load data for specified vehicle ID(s) and road ID using pandas
    
    Args:
        vehicle_id: Single vehicle ID to load (for single mode)
        road_id: Road intersection ID 
        vehicle_count: Number of vehicles to load from the beginning (for batch mode)
    """
    try:
        # Build CSV file path based on road_id
        csv_file_path = os.path.join(CSV_BASE_PATH, f'{road_id}.csv')
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Filter data based on mode
        if vehicle_count is not None:
            # Batch mode: get vehicles with ID <= vehicle_count
            vehicle_data = df[df['vehicle_id'] <= vehicle_count]
        else:
            # Single mode: get specific vehicle (default to 1 if not specified)
            if vehicle_id is None:
                vehicle_id = 1
            vehicle_data = df[df['vehicle_id'] == vehicle_id]
        
        # Convert to list of dictionaries format
        data_list = []
        for _, row in vehicle_data.iterrows():
            data_point = {
                'vehicle_id': int(row['vehicle_id']),
                'collectiontime': int(row['collectiontime']),
                'time_stamp': str(row['time_stamp']),
                'road_id': str(row['road_id']),
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'speed': float(row['speed']),
                'acceleratorpedal': float(row['acceleratorpedal']),
                'brakestatus': int(row['brakestatus'])
            }
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
            
            # Get data point limit, default to return all data
            limit = params.get('limit')
            if limit:
                limit = int(limit)
            
            # Check if this is batch mode or single mode
            vehicle_count = params.get('vehicle_count')
            if vehicle_count:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count)
                data = load_vehicle_data(road_id=road_id, vehicle_count=vehicle_count)
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                data = load_vehicle_data(vehicle_id=vehicle_id, road_id=road_id)
            
            if limit and limit > 0:
                data = data[:limit]
            
            # Build response based on mode
            if vehicle_count:
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
            
            # Check if this is batch mode or single mode
            vehicle_count = params.get('vehicle_count')
            if vehicle_count:
                # Batch mode: load multiple vehicles
                vehicle_count = int(vehicle_count)
                data = load_vehicle_data(road_id=road_id, vehicle_count=vehicle_count)
                vehicle_id = None  # Not applicable in batch mode
            else:
                # Single mode: load specific vehicle
                vehicle_id = int(params.get('vehicle_id', 1))
                data = load_vehicle_data(vehicle_id=vehicle_id, road_id=road_id)
            
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