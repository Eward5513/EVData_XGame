#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# CSV file path
CSV_FILE_PATH = '../../A0003.csv'

def load_vehicle_data(vehicle_id=1):
    """
    Load data for specified vehicle ID
    """
    try:
        # Read CSV file
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Filter data for specified vehicle ID
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

@app.route('/api/vehicle/data', methods=['GET'])
def get_vehicle_data():
    """
    API endpoint to get vehicle data
    """
    try:
        # Get vehicle ID from query parameters, default to 1
        vehicle_id = request.args.get('vehicle_id', 1, type=int)
        
        # Get data point limit, default to return all data
        limit = request.args.get('limit', None, type=int)
        
        # Load data
        data = load_vehicle_data(vehicle_id)
        
        if limit and limit > 0:
            data = data[:limit]
        
        response = {
            'status': 'success',
            'vehicle_id': vehicle_id,
            'total_points': len(data),
            'data': data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/vehicle/summary', methods=['GET'])
def get_vehicle_summary():
    """
    Get vehicle data summary information
    """
    try:
        vehicle_id = request.args.get('vehicle_id', 1, type=int)
        data = load_vehicle_data(vehicle_id)
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data found'
            }), 404
        
        # Calculate summary information
        longitudes = [point['longitude'] for point in data]
        latitudes = [point['latitude'] for point in data]
        speeds = [point['speed'] for point in data]
        
        summary = {
            'status': 'success',
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
        
        return jsonify(summary)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running normally'
    })

@app.route('/', methods=['GET'])
def index():
    """
    Home page
    """
    return jsonify({
        'message': 'Geographic Data Visualization Backend Service',
        'version': '1.0.0',
        'endpoints': [
            '/api/vehicle/data - Get vehicle trajectory data',
            '/api/vehicle/summary - Get vehicle data summary',
            '/health - Health check'
        ]
    })

if __name__ == '__main__':
    print("Starting Geographic Data Visualization Backend Service...")
    print(f"CSV file path: {os.path.abspath(CSV_FILE_PATH)}")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Warning: CSV file {CSV_FILE_PATH} does not exist")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
