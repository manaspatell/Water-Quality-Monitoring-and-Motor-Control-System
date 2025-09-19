from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from datetime import datetime
import json
from ai_models import MotorAISystem, WaterAISystem

app = Flask(__name__)

# Global variables
water_model = None
water_scaler = None
water_means = None
potable_means = None
motor_model = None
motor_scaler = None
motor_means = None
healthy_means = None

# Alert system
alerts = []
alert_thresholds = {
    'temp_critical': 80,
    'temp_warning': 70,
    'vibration_critical': 6.0,
    'vibration_warning': 4.5,
    'voltage_min': 200,
    'voltage_max': 250,
    'noise_critical': 90,
    'noise_warning': 85,
    # Water thresholds (basic)
    'ph_min': 6.5,
    'ph_max': 8.5,
    'hardness_max': 500,
    'sulfate_max': 500,
    'turbidity_max': 10
}

# AI Systems
ai_system = MotorAISystem()
water_ai_system = WaterAISystem()

# Motor Control System
motor_state = {
    'is_running': True,
    'shutdown_reason': None,
    'shutdown_time': None,
    'last_critical_alert': None,
    'auto_shutdown_enabled': True
}

def check_critical_shutdown_conditions(temp, vibration, voltage, noise, prediction):
    """Check if motor should be automatically shut down due to critical conditions"""
    global motor_state
    
    critical_conditions = []
    shutdown_reason = None
    detailed_conditions = []
    
    # Check each condition individually and collect details
    temp_critical = temp >= 90
    vib_critical = vibration >= 8.0
    volt_critical = voltage < 180 or voltage > 280
    noise_critical = noise >= 100
    ml_critical = prediction == 0 and temp >= 85
    
    # Collect detailed condition information
    if temp_critical:
        detailed_conditions.append(f"Temperature: {temp}°C (Limit: 90°C)")
    if vib_critical:
        detailed_conditions.append(f"Vibration: {vibration}mm/s (Limit: 8.0mm/s)")
    if volt_critical:
        detailed_conditions.append(f"Voltage: {voltage}V (Range: 180-280V)")
    if noise_critical:
        detailed_conditions.append(f"Noise: {noise}dB (Limit: 100dB)")
    if ml_critical:
        detailed_conditions.append(f"ML Fault Prediction with High Temperature: {temp}°C")
    
    # Single critical condition shutdowns (most severe)
    if temp_critical:
        critical_conditions.append('extreme_overheating')
        shutdown_reason = f"CRITICAL: Motor temperature {temp}°C exceeds extreme limit (90°C) - Automatic shutdown initiated"
    
    elif vib_critical:
        critical_conditions.append('extreme_vibration')
        shutdown_reason = f"CRITICAL: Motor vibration {vibration}mm/s exceeds extreme limit (8.0mm/s) - Automatic shutdown initiated"
    
    elif volt_critical:
        critical_conditions.append('extreme_voltage')
        shutdown_reason = f"CRITICAL: Motor voltage {voltage}V outside extreme range (180-280V) - Automatic shutdown initiated"
    
    elif noise_critical:
        critical_conditions.append('extreme_noise')
        shutdown_reason = f"CRITICAL: Motor noise {noise}dB exceeds extreme limit (100dB) - Automatic shutdown initiated"
    
    elif ml_critical:
        critical_conditions.append('ml_critical_fault')
        shutdown_reason = f"CRITICAL: ML model predicts critical fault with temperature {temp}°C - Automatic shutdown initiated"
    
    # Multiple critical conditions (less severe but still dangerous)
    # Only check this if no single extreme condition was met
    elif not critical_conditions:  # Only if no single extreme condition was triggered
        multiple_conditions = []
        if temp >= 85:
            multiple_conditions.append(f"Temperature: {temp}°C (Warning: ≥85°C)")
        if vibration >= 6.5:
            multiple_conditions.append(f"Vibration: {vibration}mm/s (Warning: ≥6.5mm/s)")
        if voltage < 190 or voltage > 270:
            multiple_conditions.append(f"Voltage: {voltage}V (Warning: <190V or >270V)")
        if noise >= 95:
            multiple_conditions.append(f"Noise: {noise}dB (Warning: ≥95dB)")
        
        if len(multiple_conditions) >= 2:
            critical_conditions.append('multiple_critical')
            detailed_conditions.extend(multiple_conditions)
            shutdown_reason = f"CRITICAL: Multiple critical conditions detected - Automatic shutdown initiated"
    
    # Execute shutdown if critical conditions are met
    if critical_conditions and motor_state['auto_shutdown_enabled'] and motor_state['is_running']:
        motor_state['is_running'] = False
        motor_state['shutdown_reason'] = shutdown_reason
        motor_state['shutdown_time'] = datetime.now().isoformat()
        motor_state['last_critical_alert'] = {
            'conditions': critical_conditions,
            'detailed_conditions': detailed_conditions,
            'reason': shutdown_reason,
            'timestamp': motor_state['shutdown_time'],
            'sensor_data': {
                'temperature': temp,
                'vibration': vibration,
                'voltage': voltage,
                'noise': noise
            }
        }
        
        print(f"🚨 AUTOMATIC MOTOR SHUTDOWN: {shutdown_reason}")
        print(f"📊 Detailed Conditions: {', '.join(detailed_conditions)}")
        return True, shutdown_reason, critical_conditions, detailed_conditions
    
    return False, None, [], []

def check_motor_alerts(temp, vibration, voltage, noise, prediction):
    """Check for motor alerts based on thresholds and ML prediction"""
    new_alerts = []
    current_time = datetime.now().isoformat()
    
    # Temperature alerts
    if temp >= alert_thresholds['temp_critical']:
        new_alerts.append({
            'id': f"temp_critical_{current_time}",
            'type': 'critical',
            'category': 'temperature',
            'message': f'CRITICAL: Temperature {temp}°C exceeds critical threshold ({alert_thresholds["temp_critical"]}°C)',
            'timestamp': current_time,
            'value': temp,
            'threshold': alert_thresholds['temp_critical']
        })
    elif temp >= alert_thresholds['temp_warning']:
        new_alerts.append({
            'id': f"temp_warning_{current_time}",
            'type': 'warning',
            'category': 'temperature',
            'message': f'WARNING: Temperature {temp}°C exceeds warning threshold ({alert_thresholds["temp_warning"]}°C)',
            'timestamp': current_time,
            'value': temp,
            'threshold': alert_thresholds['temp_warning']
        })
    
    # Vibration alerts
    if vibration >= alert_thresholds['vibration_critical']:
        new_alerts.append({
            'id': f"vib_critical_{current_time}",
            'type': 'critical',
            'category': 'vibration',
            'message': f'CRITICAL: Vibration {vibration}mm/s exceeds critical threshold ({alert_thresholds["vibration_critical"]}mm/s)',
            'timestamp': current_time,
            'value': vibration,
            'threshold': alert_thresholds['vibration_critical']
        })
    elif vibration >= alert_thresholds['vibration_warning']:
        new_alerts.append({
            'id': f"vib_warning_{current_time}",
            'type': 'warning',
            'category': 'vibration',
            'message': f'WARNING: Vibration {vibration}mm/s exceeds warning threshold ({alert_thresholds["vibration_warning"]}mm/s)',
            'timestamp': current_time,
            'value': vibration,
            'threshold': alert_thresholds['vibration_warning']
        })
    
    # Voltage alerts
    if voltage < alert_thresholds['voltage_min'] or voltage > alert_thresholds['voltage_max']:
        new_alerts.append({
            'id': f"voltage_{current_time}",
            'type': 'warning',
            'category': 'voltage',
            'message': f'WARNING: Voltage {voltage}V is outside normal range ({alert_thresholds["voltage_min"]}-{alert_thresholds["voltage_max"]}V)',
            'timestamp': current_time,
            'value': voltage,
            'threshold': f"{alert_thresholds['voltage_min']}-{alert_thresholds['voltage_max']}"
        })
    
    # Noise alerts
    if noise >= alert_thresholds['noise_critical']:
        new_alerts.append({
            'id': f"noise_critical_{current_time}",
            'type': 'critical',
            'category': 'noise',
            'message': f'CRITICAL: Noise {noise}dB exceeds critical threshold ({alert_thresholds["noise_critical"]}dB)',
            'timestamp': current_time,
            'value': noise,
            'threshold': alert_thresholds['noise_critical']
        })
    elif noise >= alert_thresholds['noise_warning']:
        new_alerts.append({
            'id': f"noise_warning_{current_time}",
            'type': 'warning',
            'category': 'noise',
            'message': f'WARNING: Noise {noise}dB exceeds warning threshold ({alert_thresholds["noise_warning"]}dB)',
            'timestamp': current_time,
            'value': noise,
            'threshold': alert_thresholds['noise_warning']
        })
    
    # ML Model prediction alert
    if prediction == 0:  # Fault predicted
        new_alerts.append({
            'id': f"ml_fault_{current_time}",
            'type': 'critical',
            'category': 'ml_prediction',
            'message': 'CRITICAL: ML Model predicts motor fault - immediate attention required',
            'timestamp': current_time,
            'value': 'Fault Predicted',
            'threshold': 'N/A'
        })
    
    # Add new alerts to global alerts list
    for alert in new_alerts:
        alerts.append(alert)
    
    # Keep only last 100 alerts to prevent memory issues
    if len(alerts) > 100:
        alerts[:] = alerts[-100:]
    
    return new_alerts

def check_water_alerts(ph, hardness, sulfate, turbidity, prediction, ai_insights):
    """Check for water alerts based on thresholds and AI prediction"""
    new_alerts = []
    current_time = datetime.now().isoformat()

    # Parameter alerts
    if ph < alert_thresholds['ph_min'] or ph > alert_thresholds['ph_max']:
        new_alerts.append({
            'id': f"ph_alert_{current_time}",
            'type': 'warning',
            'category': 'water_ph',
            'message': f"pH {ph} outside safe range ({alert_thresholds['ph_min']}-{alert_thresholds['ph_max']})",
            'timestamp': current_time,
            'value': ph,
            'threshold': f"{alert_thresholds['ph_min']}-{alert_thresholds['ph_max']}"
        })
    if hardness > alert_thresholds['hardness_max']:
        new_alerts.append({
            'id': f"hardness_alert_{current_time}",
            'type': 'warning',
            'category': 'water_hardness',
            'message': f"Hardness {hardness}mg/L is high (>{alert_thresholds['hardness_max']}mg/L)",
            'timestamp': current_time,
            'value': hardness,
            'threshold': alert_thresholds['hardness_max']
        })
    if sulfate > alert_thresholds['sulfate_max']:
        new_alerts.append({
            'id': f"sulfate_alert_{current_time}",
            'type': 'warning',
            'category': 'water_sulfate',
            'message': f"Sulfate {sulfate}mg/L is high (>{alert_thresholds['sulfate_max']}mg/L)",
            'timestamp': current_time,
            'value': sulfate,
            'threshold': alert_thresholds['sulfate_max']
        })
    if turbidity > alert_thresholds['turbidity_max']:
        new_alerts.append({
            'id': f"turbidity_alert_{current_time}",
            'type': 'warning',
            'category': 'water_turbidity',
            'message': f"Turbidity {turbidity}NTU is high (>{alert_thresholds['turbidity_max']}NTU)",
            'timestamp': current_time,
            'value': turbidity,
            'threshold': alert_thresholds['turbidity_max']
        })

    # Do-not-drink critical alert
    risk = 1.0
    try:
        risk = ai_insights.get('failure_prediction', {}).get('failure_probability', 1.0)
    except Exception:
        pass
    if int(prediction) == 0 or risk >= 0.6 or turbidity > 10 or ph < 6.0 or ph > 9.0:
        new_alerts.append({
            'id': f"do_not_drink_{current_time}",
            'type': 'critical',
            'category': 'water_quality',
            'message': 'CRITICAL: Do NOT drink this water without treatment',
            'timestamp': current_time,
            'value': 'Not Potable',
            'threshold': 'N/A'
        })

    # Add to global alerts but cap length
    alerts.extend(new_alerts)
    if len(alerts) > 100:
        alerts[:] = alerts[-100:]

    return new_alerts

def initialize_models():
    global water_model, water_scaler, water_means, potable_means
    global motor_model, motor_scaler, motor_means, healthy_means

    print("Initializing models...")

    try:
        df = pd.read_csv("water_potability.csv")

        # Use KNN imputer
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        X = df_imputed.drop("Potability", axis=1)
        y = df_imputed["Potability"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Standard scaling
        water_scaler = StandardScaler()
        X_train_scaled = water_scaler.fit_transform(X_train)
        X_test_scaled = water_scaler.transform(X_test)

        # XGBoost with improved parameters
        water_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1
        )
        water_model.fit(X_train_scaled, y_train)

        # Print accuracy
        preds = water_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"Water model accuracy: {acc:.4f}")

        # Means for comparison
        water_means = df_imputed.mean()
        potable_means = df_imputed[df_imputed["Potability"] == 1].mean()

        # Motor model (demo, same as water)
        motor_scaler = StandardScaler()
        motor_scaler.fit(X)
        motor_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        motor_model.fit(water_scaler.transform(X), y)
        motor_means = df_imputed.mean()
        healthy_means = potable_means

        # Train AI models
        print("🤖 Training AI Models...")
        ai_system.create_enhanced_models(X, y)
        ai_system.create_anomaly_detector(X)
        ai_system.save_models()

        # Train Water AI models
        print("💧 Training Water AI Models...")
        water_ai_system.create_enhanced_models(X, y)
        water_ai_system.create_anomaly_detector(X)
        water_ai_system.save_models()

        print("Models initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return False

@app.before_request
def before_first_request():
    if water_model is None:
        initialize_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/water')
def water():
    return render_template('index.html')

@app.route('/motor')
def motor():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html')

@app.route('/predict_water', methods=['POST'])
def predict_water():
    if water_model is None:
        return jsonify({'error': 'Models not initialized'}), 500

    try:
        data = request.json

        user_input = [
            float(data['ph']),
            float(data['hardness']),
            water_means["Solids"],
            water_means["Chloramines"],
            float(data['sulfate']),
            water_means["Conductivity"],
            water_means["Organic_carbon"],
            water_means["Trihalomethanes"],
            float(data['turbidity'])
        ]

        input_scaled = water_scaler.transform([user_input])
        prediction = water_model.predict(input_scaled)[0]

        # Get AI insights for water quality
        sensor_data = [
            float(data['ph']),
            float(data['hardness']),
            float(data['sulfate']),
            float(data['turbidity'])
        ]
        ai_insights = water_ai_system.get_ai_insights(sensor_data)

        # Determine suggested uses
        suggested_uses = []
        if 6.5 <= sensor_data[0] <= 8.5 and sensor_data[1] < 200 and sensor_data[2] < 250 and sensor_data[3] < 5:
            suggested_uses.append('Drinking')
        if 6.0 <= sensor_data[0] <= 9.0 and sensor_data[1] < 500 and sensor_data[2] < 500 and sensor_data[3] < 10:
            suggested_uses.append('Household')
        if 5.5 <= sensor_data[0] <= 8.5 and sensor_data[1] < 1000 and sensor_data[2] < 1000 and sensor_data[3] < 15:
            suggested_uses.append('Irrigation')
        if not suggested_uses:
            suggested_uses.append('Unusable without treatment')

        # Water alerts
        new_alerts = check_water_alerts(sensor_data[0], sensor_data[1], sensor_data[2], sensor_data[3], prediction, ai_insights)

        return jsonify({
            'prediction': int(prediction),
            'user_vals': [
                float(data['ph']),
                float(data['hardness']),
                float(data['sulfate']),
                float(data['turbidity'])
            ],
            'ideal_vals': [
                potable_means["ph"],
                potable_means["Hardness"],
                potable_means["Sulfate"],
                potable_means["Turbidity"]
            ],
            'ai_insights': ai_insights,
            'alerts': new_alerts,
            'suggested_uses': suggested_uses
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_motor', methods=['POST'])
def predict_motor():
    if motor_model is None:
        return jsonify({'error': 'Models not initialized'}), 500

    try:
        data = request.json

        user_input = [
            float(data['temp']),
            float(data['vibration']),
            motor_means["Solids"],
            motor_means["Chloramines"],
            float(data['voltage']),
            motor_means["Conductivity"],
            motor_means["Organic_carbon"],
            motor_means["Trihalomethanes"],
            float(data['noise'])
        ]

        input_scaled = motor_scaler.transform([user_input])
        prediction = motor_model.predict(input_scaled)[0]

        # Check for critical shutdown conditions first
        shutdown_triggered, shutdown_reason, critical_conditions, detailed_conditions = check_critical_shutdown_conditions(
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise']),
            int(prediction)
        )

        # Check for alerts
        new_alerts = check_motor_alerts(
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise']),
            int(prediction)
        )

        # Add shutdown alert if triggered
        if shutdown_triggered:
            shutdown_alert = {
                'id': f"motor_shutdown_{datetime.now().isoformat()}",
                'type': 'critical',
                'category': 'motor_shutdown',
                'message': shutdown_reason,
                'timestamp': motor_state['shutdown_time'],
                'value': 'Motor Shutdown',
                'threshold': 'N/A',
                'critical_conditions': critical_conditions,
                'detailed_conditions': detailed_conditions
            }
            new_alerts.append(shutdown_alert)
            alerts.append(shutdown_alert)

        # Get AI insights
        sensor_data = [float(data['temp']), float(data['vibration']), 
                      float(data['voltage']), float(data['noise'])]
        ai_insights = ai_system.get_ai_insights(sensor_data)

        return jsonify({
            'prediction': int(prediction),
            'motor_state': motor_state,
            'shutdown_triggered': shutdown_triggered,
            'shutdown_reason': shutdown_reason,
            'critical_conditions': critical_conditions,
            'detailed_conditions': detailed_conditions,
            'user_vals': [
                float(data['temp']),
                float(data['vibration']),
                float(data['voltage']),
                float(data['noise'])
            ],
            'ideal_vals': [
                healthy_means["ph"],
                healthy_means["Hardness"],
                healthy_means["Sulfate"],
                healthy_means["Turbidity"]
            ],
            'alerts': new_alerts,
            'ai_insights': ai_insights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/motor/control', methods=['GET', 'POST'])
def motor_control():
    """Get motor state or control motor (start/stop)"""
    global motor_state
    
    if request.method == 'GET':
        return jsonify({
            'motor_state': motor_state,
            'status': 'shutdown' if not motor_state['is_running'] else 'running',
            'message': motor_state['shutdown_reason'] if not motor_state['is_running'] else 'Motor is running normally'
        })
    
    elif request.method == 'POST':
        data = request.json
        action = data.get('action', '').lower()
        
        if action == 'start':
            if not motor_state['is_running']:
                motor_state['is_running'] = True
                motor_state['shutdown_reason'] = None
                motor_state['shutdown_time'] = None
                motor_state['last_critical_alert'] = None
                return jsonify({
                    'success': True,
                    'message': 'Motor started successfully',
                    'motor_state': motor_state
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Motor is already running',
                    'motor_state': motor_state
                })
        
        elif action == 'stop':
            if motor_state['is_running']:
                motor_state['is_running'] = False
                motor_state['shutdown_reason'] = 'Manual shutdown initiated by operator'
                motor_state['shutdown_time'] = datetime.now().isoformat()
                return jsonify({
                    'success': True,
                    'message': 'Motor stopped successfully',
                    'motor_state': motor_state
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Motor is already stopped',
                    'motor_state': motor_state
                })
        
        elif action == 'toggle_auto_shutdown':
            motor_state['auto_shutdown_enabled'] = not motor_state['auto_shutdown_enabled']
            return jsonify({
                'success': True,
                'message': f'Auto-shutdown {"enabled" if motor_state["auto_shutdown_enabled"] else "disabled"}',
                'motor_state': motor_state
            })
        
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid action. Use: start, stop, or toggle_auto_shutdown'
            }), 400

@app.route('/motor/status')
def motor_status():
    """Get detailed motor status"""
    global motor_state
    
    return jsonify({
        'motor_state': motor_state,
        'is_running': motor_state['is_running'],
        'auto_shutdown_enabled': motor_state['auto_shutdown_enabled'],
        'shutdown_info': {
            'reason': motor_state['shutdown_reason'],
            'time': motor_state['shutdown_time'],
            'last_critical_alert': motor_state['last_critical_alert']
        } if not motor_state['is_running'] else None,
        'status_message': motor_state['shutdown_reason'] if not motor_state['is_running'] else 'Motor is running normally'
    })

@app.route('/alerts')
def get_alerts():
    """Get all alerts"""
    return jsonify({
        'alerts': alerts,
        'total': len(alerts),
        'critical_count': len([a for a in alerts if a['type'] == 'critical']),
        'warning_count': len([a for a in alerts if a['type'] == 'warning'])
    })

@app.route('/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    global alerts
    alerts.clear()
    return jsonify({'message': 'All alerts cleared', 'count': 0})

@app.route('/alerts/acknowledge/<alert_id>', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge a specific alert"""
    for alert in alerts:
        if alert['id'] == alert_id:
            alert['acknowledged'] = True
            alert['acknowledged_at'] = datetime.now().isoformat()
            return jsonify({'message': 'Alert acknowledged', 'alert_id': alert_id})
    return jsonify({'error': 'Alert not found'}), 404

@app.route('/alerts/thresholds', methods=['GET', 'POST'])
def manage_thresholds():
    """Get or update alert thresholds"""
    global alert_thresholds
    
    if request.method == 'GET':
        return jsonify(alert_thresholds)
    
    elif request.method == 'POST':
        data = request.json
        for key, value in data.items():
            if key in alert_thresholds:
                alert_thresholds[key] = float(value)
        return jsonify({
            'message': 'Thresholds updated',
            'thresholds': alert_thresholds
        })

@app.route('/ai/predict', methods=['POST'])
def ai_predict():
    """Get AI predictions for motor health"""
    try:
        data = request.json
        sensor_data = [
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise'])
        ]
        
        ai_insights = ai_system.get_ai_insights(sensor_data)
        return jsonify(ai_insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ai/maintenance', methods=['POST'])
def ai_maintenance():
    """Get AI maintenance recommendations"""
    try:
        data = request.json
        sensor_data = [
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise'])
        ]
        
        maintenance_pred = ai_system.predict_maintenance_schedule(sensor_data)
        return jsonify(maintenance_pred)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ai/anomaly', methods=['POST'])
def ai_anomaly():
    """Detect anomalies in sensor data"""
    try:
        data = request.json
        sensor_data = [
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise'])
        ]
        
        anomaly_result = ai_system.detect_anomalies(sensor_data)
        return jsonify(anomaly_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ai/health_score', methods=['POST'])
def ai_health_score():
    """Get AI health score"""
    try:
        data = request.json
        sensor_data = [
            float(data['temp']),
            float(data['vibration']),
            float(data['voltage']),
            float(data['noise'])
        ]
        
        health_score = ai_system._calculate_health_score(sensor_data)
        return jsonify({'health_score': health_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.cli.command()
def init_models():
    if initialize_models():
        print("Models initialized successfully")
    else:
        print("Failed to initialize models")

if __name__ == '__main__':
    if initialize_models():
        app.run(debug=True)
    else:
        print("Failed to start application due to model initialization error")
