import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MotorAISystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.anomaly_detector = None
        self.prediction_history = []
        self.failure_probability = 0.0
        self.maintenance_recommendation = "Normal Operation"

    def create_enhanced_models(self, X, y):
        print("🤖 Training AI Models...")
        motor_y = y.copy() if X.shape[1] == 9 else y
        X_train, X_test, y_train, y_test = train_test_split(X, motor_y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf

        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        nn.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn

        ensemble = VotingClassifier(estimators=[('rf', rf), ('nn', nn)], voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        self.models['ensemble'] = ensemble

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ {name.replace('_', ' ').title()}: {accuracy:.4f}")
        return self.models

    def create_anomaly_detector(self, X):
        print("🔍 Training Anomaly Detection Model...")
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['anomaly'] = scaler
        self.anomaly_detector.fit(X_scaled)
        print("✅ Anomaly Detection Model Trained")
        return self.anomaly_detector

    def predict_failure_probability(self, sensor_data):
        if not self.models:
            return {"error": "Models not trained"}
        X = np.array([sensor_data + [0] * 5]).reshape(1, -1) if len(sensor_data) == 4 else np.array([sensor_data]).reshape(1, -1)
        X_scaled = self.scalers['main'].transform(X)
        predictions, probabilities = {}, {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            predictions[name] = pred
            probabilities[name] = prob[1] if len(prob) > 1 else prob[0]
        ensemble_prob = np.mean(list(probabilities.values()))
        if len(sensor_data) == 4:
            ensemble_prob = self._calculate_motor_failure_probability(sensor_data)
        record = {
            'timestamp': datetime.now().isoformat(),
            'sensor_data': sensor_data,
            'ensemble_probability': ensemble_prob,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
        self.prediction_history.append(record)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        self.failure_probability = ensemble_prob
        return {
            'failure_probability': float(ensemble_prob),
            'risk_level': self._get_risk_level(ensemble_prob),
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'recommendation': self._get_recommendation(ensemble_prob)
        }

    def detect_anomalies(self, sensor_data):
        if not self.anomaly_detector:
            return {"error": "Anomaly detector not trained"}
        X = np.array([sensor_data + [0] * 5]).reshape(1, -1) if len(sensor_data) == 4 else np.array([sensor_data]).reshape(1, -1)
        X_scaled = self.scalers['anomaly'].transform(X)
        anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
        if len(sensor_data) == 4:
            temp, vibration, voltage, noise = sensor_data
            is_anomaly = (
                temp > 85 or temp < 30 or
                vibration > 7 or vibration < 0.5 or
                voltage < 180 or voltage > 280 or
                noise > 100 or noise < 40
            )
            anomaly_score = -0.5 if is_anomaly else 0.5
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'severity': self._get_anomaly_severity(anomaly_score)
        }

    def predict_maintenance_schedule(self, sensor_data, days_ahead=30):
        if not self.models:
            return {"error": "Models not trained"}
        current_prob = (
            self._calculate_motor_failure_probability(sensor_data)
            if len(sensor_data) == 4
            else self.predict_failure_probability(sensor_data)['failure_probability']
        )
        temp, vibration, voltage, noise = sensor_data
        base_interval = 30
        if current_prob < 0.1:
            next_maintenance = base_interval + 15
        elif current_prob < 0.3:
            next_maintenance = base_interval
        elif current_prob < 0.5:
            next_maintenance = base_interval - 10
        elif current_prob < 0.7:
            next_maintenance = base_interval - 20
        else:
            next_maintenance = max(1, base_interval - 25)
        if temp > 75:
            next_maintenance = max(1, next_maintenance - 5)
        if vibration > 5:
            next_maintenance = max(1, next_maintenance - 3)
        if voltage < 200 or voltage > 250:
            next_maintenance = max(1, next_maintenance - 2)
        if noise > 85:
            next_maintenance = max(1, next_maintenance - 1)
        return {
            'next_maintenance_days': max(1, next_maintenance),
            'urgency': self._get_maintenance_urgency(next_maintenance),
            'recommended_actions': self._get_maintenance_actions(sensor_data, current_prob)
        }

    def get_ai_insights(self, sensor_data):
        if len(sensor_data) == 4:
            motor_failure_prob = self._calculate_motor_failure_probability(sensor_data)
            failure_pred = {
                'failure_probability': motor_failure_prob,
                'risk_level': self._get_risk_level(motor_failure_prob),
                'recommendation': self._get_recommendation(motor_failure_prob)
            }
        else:
            failure_pred = self.predict_failure_probability(sensor_data)
        anomaly_pred = self.detect_anomalies(sensor_data)
        maintenance_pred = self.predict_maintenance_schedule(sensor_data)
        trend_analysis = self._analyze_trends()
        return {
            'failure_prediction': failure_pred,
            'anomaly_detection': anomaly_pred,
            'maintenance_schedule': maintenance_pred,
            'trend_analysis': trend_analysis,
            'overall_health_score': self._calculate_health_score(sensor_data),
            'ai_recommendations': self._generate_recommendations(sensor_data, failure_pred, anomaly_pred)
        }

    def _get_risk_level(self, probability):
        if probability < 0.1:
            return "LOW"
        elif probability < 0.3:
            return "MEDIUM"
        elif probability < 0.6:
            return "HIGH"
        else:
            return "CRITICAL"

    def _get_anomaly_severity(self, score):
        if score > -0.5:
            return "LOW"
        elif score > -1.0:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_recommendation(self, probability):
        if probability < 0.3:
            return "Continue normal operation"
        elif probability < 0.6:
            return "Monitor closely, schedule inspection"
        elif probability < 0.8:
            return "Schedule maintenance soon"
        else:
            return "Immediate maintenance required"

    def _get_maintenance_urgency(self, days):
        if days <= 7:
            return "URGENT"
        elif days <= 14:
            return "HIGH"
        elif days <= 30:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_maintenance_actions(self, sensor_data, probability):
        actions = []
        temp, vibration, voltage, noise = sensor_data
        if temp > 70:
            actions.append("Check cooling system and clean filters")
        if vibration > 4.5:
            actions.append("Inspect bearings and alignment")
        if voltage < 200 or voltage > 250:
            actions.append("Check electrical connections and power supply")
        if noise > 85:
            actions.append("Inspect for mechanical wear and lubrication")
        if probability > 0.7:
            actions.append("Perform comprehensive motor inspection")
            actions.append("Check all safety systems")
        return actions if actions else ["Routine maintenance check"]

    def _generate_recommendations(self, sensor_data, failure_pred, anomaly_pred):
        temp, vibration, voltage, noise = sensor_data
        recs = []
        # Condition-based suggestions
        if temp >= 80:
            recs.append("Reduce load and inspect cooling; verify ambient ventilation")
        elif temp >= 70:
            recs.append("Monitor temperature; clean air filters and check coolant path")
        if vibration >= 6.0:
            recs.append("Immediate bearing inspection and shaft alignment check")
        elif vibration >= 4.5:
            recs.append("Schedule vibration analysis; check mounts and imbalance")
        if voltage < 200 or voltage > 250:
            recs.append("Stabilize supply; inspect power connections and transformer taps")
        if noise >= 90:
            recs.append("Investigate for mechanical wear; verify lubrication schedule")

        # Model-driven suggestions
        risk = failure_pred.get('failure_probability', 0.0)
        risk_level = failure_pred.get('risk_level', 'LOW')
        if risk_level in ("HIGH", "CRITICAL") or risk >= 0.6:
            recs.append("Plan downtime and perform comprehensive preventive maintenance")
        elif risk >= 0.3:
            recs.append("Increase monitoring frequency and schedule inspection")

        if anomaly_pred.get('is_anomaly'):
            recs.append("Review latest process changes; validate sensor calibration")

        # Ensure at least one recommendation
        if not recs:
            recs.append("Parameters nominal; continue normal operation and routine checks")
        return recs

    def _analyze_trends(self):
        if len(self.prediction_history) < 5:
            return {"status": "Insufficient data for trend analysis"}
        recent_probs = [p['ensemble_probability'] for p in self.prediction_history[-10:]]
        trend = "stable"
        if len(recent_probs) >= 3:
            if recent_probs[-1] > recent_probs[0] + 0.1:
                trend = "increasing"
            elif recent_probs[-1] < recent_probs[0] - 0.1:
                trend = "decreasing"
        return {
            'trend': trend,
            'average_probability': np.mean(recent_probs),
            'data_points': len(recent_probs)
        }

    def _calculate_health_score(self, sensor_data):
        temp, vibration, voltage, noise = sensor_data
        if temp <= 60:
            temp_score = 100
        elif temp <= 70:
            temp_score = 95
        elif temp <= 80:
            temp_score = 85
        elif temp <= 90:
            temp_score = 70
        else:
            temp_score = max(0, 100 - (temp - 90) * 3)
        if vibration <= 3:
            vib_score = 100
        elif vibration <= 4:
            vib_score = 95
        elif vibration <= 5:
            vib_score = 85
        elif vibration <= 6:
            vib_score = 70
        else:
            vib_score = max(0, 100 - (vibration - 6) * 8)
        if 210 <= voltage <= 250:
            volt_score = 100
        elif 200 <= voltage <= 260:
            volt_score = 90
        elif 190 <= voltage <= 270:
            volt_score = 70
        else:
            volt_score = 40
        if noise <= 80:
            noise_score = 100
        elif noise <= 85:
            noise_score = 95
        elif noise <= 90:
            noise_score = 85
        elif noise <= 95:
            noise_score = 70
        else:
            noise_score = max(0, 100 - (noise - 95) * 2)
        return min(100, max(0, temp_score * 0.3 + vib_score * 0.3 + volt_score * 0.2 + noise_score * 0.2))

    def _calculate_motor_failure_probability(self, sensor_data):
        temp, vibration, voltage, noise = sensor_data
        failure_prob = 0.0
        if temp > 90:
            failure_prob += 0.4
        elif temp > 80:
            failure_prob += 0.3
        elif temp > 70:
            failure_prob += 0.1
        elif temp > 60:
            failure_prob += 0.05
        if vibration > 8:
            failure_prob += 0.3
        elif vibration > 6:
            failure_prob += 0.2
        elif vibration > 5:
            failure_prob += 0.1
        elif vibration > 4:
            failure_prob += 0.05
        if voltage < 190 or voltage > 270:
            failure_prob += 0.2
        elif voltage < 200 or voltage > 260:
            failure_prob += 0.1
        elif voltage < 210 or voltage > 250:
            failure_prob += 0.05
        if noise > 95:
            failure_prob += 0.1
        elif noise > 90:
            failure_prob += 0.05
        elif noise > 85:
            failure_prob += 0.02
        return min(1.0, failure_prob)

    def save_models(self, filepath="ai_models.pkl"):
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'anomaly_detector': self.anomaly_detector
        }
        joblib.dump(model_data, filepath)
        print(f"✅ AI models saved to {filepath}")

    def load_models(self, filepath="ai_models.pkl"):
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.anomaly_detector = model_data['anomaly_detector']
            print(f"✅ AI models loaded from {filepath}")
            return True
        return False


class WaterAISystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.anomaly_detector = None
        self.prediction_history = []

    def create_enhanced_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf

        nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
        nn.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn

        ensemble = VotingClassifier(estimators=[('rf', rf), ('nn', nn)], voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        self.models['ensemble'] = ensemble

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ Water {name.replace('_', ' ').title()}: {acc:.4f}")
        return self.models

    def create_anomaly_detector(self, X):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['anomaly'] = scaler
        self.anomaly_detector.fit(X_scaled)
        print("✅ Water Anomaly Detection Model Trained")
        return self.anomaly_detector

    def _calculate_health_score(self, sensor_data):
        ph, hardness, sulfate, turbidity = sensor_data
        ph_score = 100 if 6.5 <= ph <= 8.5 else (80 if 6.0 <= ph <= 9.0 else 50)
        hard_score = 100 if hardness < 200 else (80 if hardness < 500 else (60 if hardness < 1000 else 40))
        sulf_score = 100 if sulfate < 250 else (80 if sulfate < 500 else (60 if sulfate < 1000 else 40))
        turb_score = 100 if turbidity < 5 else (80 if turbidity < 10 else (60 if turbidity < 15 else 40))
        return min(100, max(0, ph_score * 0.25 + hard_score * 0.25 + sulf_score * 0.25 + turb_score * 0.25))

    def _detect_anomaly_rules(self, sensor_data):
        ph, hardness, sulfate, turbidity = sensor_data
        is_anom = (ph < 5.5 or ph > 9.5 or hardness > 1000 or sulfate > 1000 or turbidity > 15)
        return {
            'is_anomaly': bool(is_anom),
            'anomaly_score': float(-0.5 if is_anom else 0.5),
            'severity': "HIGH" if is_anom else "LOW"
        }

    def _recommendations(self, sensor_data, risk_prob):
        ph, hardness, sulfate, turbidity = sensor_data
        recommendations: list[str] = []

        # pH treatment
        if ph < 6.5:
            recommendations.append("pH low: dose alkaline buffer (e.g., sodium bicarbonate) or use calcite neutralizing filter to raise pH into 6.5–8.5.")
        elif ph > 8.5:
            recommendations.append("pH high: dose dilute food-grade acid (e.g., citric acid) with control, or blend with neutral water to bring pH into 6.5–8.5.")

        # Hardness treatment
        if hardness >= 500:
            recommendations.append("Hardness very high: install ion-exchange softener (Na⁺/K⁺) or Reverse Osmosis; protect downstream equipment from scaling.")
        elif hardness >= 200:
            recommendations.append("Hardness high: consider softener or antiscalant dosing; descale kettles/heaters periodically.")

        # Sulfate treatment
        if sulfate >= 500:
            recommendations.append("Sulfate very high: apply Reverse Osmosis or Nanofiltration; investigate contamination source (industrial/agricultural).")
        elif sulfate >= 250:
            recommendations.append("Sulfate elevated: RO/NF recommended; evaluate source and reduce ingress.")

        # Turbidity treatment
        if turbidity >= 10:
            recommendations.append("Turbidity high: perform coagulation/flocculation (alum/polymer), sedimentation, and multi-media filtration; follow with disinfection.")
        elif turbidity >= 5:
            recommendations.append("Turbidity elevated: improve filtration (cartridge/UF) and ensure proper prefiltration before RO; verify filter integrity.")

        # General polishing / safety
        if any([
            ph < 6.0 or ph > 9.0,
            hardness >= 500,
            sulfate >= 500,
            turbidity >= 10,
            risk_prob >= 0.6,
        ]):
            recommendations.append("Do NOT drink without treatment: parameters outside safe range or high risk level.")

        # Suggested uses guidance
        suggested_uses: list[str] = []
        if 6.5 <= ph <= 8.5 and hardness < 200 and sulfate < 250 and turbidity < 5:
            suggested_uses.append("Suitable for Drinking (meets common guidelines)")
        if 6.0 <= ph <= 9.0 and hardness < 500 and sulfate < 500 and turbidity < 10:
            suggested_uses.append("Suitable for Household use (cleaning, bathing)")
        if 5.5 <= ph <= 8.5 and hardness < 1000 and sulfate < 1000 and turbidity < 15:
            suggested_uses.append("Suitable for Irrigation (most crops)")
        if not suggested_uses:
            suggested_uses.append("Unusable without treatment")

        recommendations.append("Suggested uses: " + ", ".join(suggested_uses))

        # Post-treatment polish
        if recommendations and "Do NOT drink" not in " ".join(recommendations):
            recommendations.append("Optional polishing: activated carbon (taste/odor), UV/chlorination (disinfection).")

        return recommendations

    def predict_non_potability_probability(self, sensor_data):
        features = sensor_data + [0] * 5 if len(sensor_data) == 4 else sensor_data
        X = np.array([features])
        if 'main' in self.scalers and self.models:
            Xs = self.scalers['main'].transform(X)
            probs = []
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    p = model.predict_proba(Xs)[0]
                    probs.append(1.0 - (p[1] if len(p) > 1 else p[0]))
            if probs:
                return float(np.mean(probs))
        score = self._calculate_health_score(sensor_data)
        return float(max(0.0, min(1.0, (100 - score) / 100.0)))

    def detect_anomalies(self, sensor_data):
        return self._detect_anomaly_rules(sensor_data)

    def predict_maintenance_schedule(self, sensor_data):
        risk_p = self.predict_non_potability_probability(sensor_data)
        if risk_p < 0.1:
            days = 45
        elif risk_p < 0.3:
            days = 30
        elif risk_p < 0.6:
            days = 20
        elif risk_p < 0.8:
            days = 10
        else:
            days = 5
        urgency = "LOW" if days >= 30 else ("MEDIUM" if days >= 20 else ("HIGH" if days >= 10 else "URGENT"))
        return {
            'next_maintenance_days': days,
            'urgency': urgency,
            'recommended_actions': self._recommendations(sensor_data, risk_p)
        }

    def get_ai_insights(self, sensor_data):
        risk_p = self.predict_non_potability_probability(sensor_data)
        if risk_p < 0.1:
            risk_level = "LOW"; recommendation = "Water quality is acceptable."
        elif risk_p < 0.3:
            risk_level = "MEDIUM"; recommendation = "Monitor parameters; consider filtration."
        elif risk_p < 0.6:
            risk_level = "HIGH"; recommendation = "Treat water before use (filtration/RO)."
        else:
            risk_level = "CRITICAL"; recommendation = "Not potable; do not use without treatment."
        anomaly_pred = self.detect_anomalies(sensor_data)
        maintenance_pred = self.predict_maintenance_schedule(sensor_data)
        health_score = self._calculate_health_score(sensor_data)
        recs = self._recommendations(sensor_data, risk_p)
        return {
            'failure_prediction': {
                'failure_probability': float(risk_p),
                'risk_level': risk_level,
                'recommendation': recommendation
            },
            'anomaly_detection': anomaly_pred,
            'maintenance_schedule': maintenance_pred,
            'overall_health_score': float(health_score),
            'ai_recommendations': recs
        }

    def save_models(self, filepath="water_ai_models.pkl"):
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'anomaly_detector': self.anomaly_detector
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Water AI models saved to {filepath}")

    def load_models(self, filepath="water_ai_models.pkl"):
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.anomaly_detector = model_data['anomaly_detector']
            print(f"✅ Water AI models loaded from {filepath}")
            return True
        return False
