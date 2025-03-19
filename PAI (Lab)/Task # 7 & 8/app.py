# app.py
from flask import Flask, render_template, request, jsonify
import os
import requests
from datetime import datetime

app = Flask(__name__)
API_KEY = os.environ.get('WEATHER_API_KEY')
BASE_URL = "http://api.openweathermap.org/data/2.5"

def get_weather_data(params, is_forecast=False):
    try:
        endpoint = "/forecast" if is_forecast else "/weather"
        url = f"{BASE_URL}{endpoint}?{'&'.join(params)}&appid={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['GET'])
def get_weather():
    unit = request.args.get('units', 'metric')  # Default to metric if not provided
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    city = request.args.get('city')
    
    params = []
    if lat and lon:
        params.extend([f"lat={lat}", f"lon={lon}"])
    elif city:
        params.append(f"q={city}")
    else:
        return jsonify({'error': 'Missing location parameters'}), 400
    
    params.append(f"units={unit}")  # Add the units parameter
    
    data = get_weather_data(params)
    if not data:
        return jsonify({'error': 'Failed to fetch weather data'}), 500
    
    if data.get('cod') != 200:
        return jsonify({'error': data.get('message', 'Unknown error')}), data.get('cod', 500)
    
    return jsonify(process_current_weather(data))

@app.route('/forecast', methods=['GET'])
def get_forecast():
    unit = request.args.get('units', 'metric')
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    city = request.args.get('city')
    
    params = []
    if lat and lon:
        params.extend([f"lat={lat}", f"lon={lon}"])
    elif city:
        params.append(f"q={city}")
    else:
        return jsonify({'error': 'Missing location parameters'}), 400
    
    params.append(f"units={unit}")
    
    data = get_weather_data(params, is_forecast=True)
    if not data:
        return jsonify({'error': 'Failed to fetch forecast data'}), 500
    
    if data.get('cod') != '200':
        return jsonify({'error': data.get('message', 'Unknown error')}), data.get('cod', 500)
    
    return jsonify(process_forecast(data))

def process_current_weather(data):
    return {
        'city': data['name'],
        'country': data['sys']['country'],
        'temp': data['main']['temp'],
        'feels_like': data['main']['feels_like'],
        'description': data['weather'][0]['description'],
        'icon': data['weather'][0]['icon'],
        'humidity': data['main']['humidity'],
        'wind': data['wind']['speed'],
        'pressure': data['main']['pressure'],
        'visibility': data.get('visibility', 0) / 1000,  # Convert to km
        'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
        'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
        'unit': '°C' if 'metric' in request.args.get('units', 'metric') else '°F'
    }

def process_forecast(data):
    daily_data = {}
    for item in data['list']:
        date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
        if date not in daily_data:
            daily_data[date] = {
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'icons': [],
                'description': item['weather'][0]['description']
            }
        else:
            daily_data[date]['temp_min'] = min(daily_data[date]['temp_min'], item['main']['temp_min'])
            daily_data[date]['temp_max'] = max(daily_data[date]['temp_max'], item['main']['temp_max'])
        daily_data[date]['icons'].append(item['weather'][0]['icon'])
    
    forecast = []
    for i, (date, values) in enumerate(daily_data.items()):
        if i >= 5:
            break
        forecast.append({
            'date': datetime.strptime(date, '%Y-%m-%d').strftime('%a, %b %d'),
            'temp_min': values['temp_min'],
            'temp_max': values['temp_max'],
            'icon': max(set(values['icons']), key=values['icons'].count),  # Most frequent icon
            'description': values['description']
        })
    return forecast

if __name__ == '__main__':
    app.run(debug=True)
print("API Key:", API_KEY)