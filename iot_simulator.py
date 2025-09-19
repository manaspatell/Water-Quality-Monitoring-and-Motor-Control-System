import random
import time
import requests
from datetime import datetime

class MotorIOTSimulator:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.running = False
        self.motor_id = "MOTOR_001"
        
        # Normal operating ranges
        self.temp_range = (45, 65)  # °C
        self.vibration_range = (1.5, 3.5)  # mm/s
        self.voltage_range = (220, 240)  # V
        self.noise_range = (65, 80)  # dB
        
        # Fault simulation parameters
        self.fault_probability = 0.05  # 5% chance of fault per reading
        self.current_fault = None
        self.fault_duration = 0
        
    def generate_normal_reading(self):
        """Generate normal motor sensor readings"""
        return {
            'temp': round(random.uniform(*self.temp_range), 1),
            'vibration': round(random.uniform(*self.vibration_range), 1),
            'voltage': round(random.uniform(*self.voltage_range), 1),
            'noise': round(random.uniform(*self.noise_range), 1)
        }
    
    def generate_fault_reading(self, fault_type):
        """Generate readings with specific fault conditions"""
        base = self.generate_normal_reading()
        
        if fault_type == "high_temp":
            base['temp'] = round(random.uniform(75, 90), 1)
        elif fault_type == "high_vibration":
            base['vibration'] = round(random.uniform(5.0, 8.0), 1)
        elif fault_type == "voltage_fluctuation":
            base['voltage'] = round(random.uniform(180, 260), 1)
        elif fault_type == "excessive_noise":
            base['noise'] = round(random.uniform(85, 95), 1)
        elif fault_type == "multiple_faults":
            base['temp'] = round(random.uniform(70, 85), 1)
            base['vibration'] = round(random.uniform(4.5, 7.0), 1)
            base['voltage'] = round(random.uniform(200, 250), 1)
            base['noise'] = round(random.uniform(80, 90), 1)
            
        return base
    
    def send_reading(self, reading):
        """Send reading to the Flask app"""
        try:
            response = requests.post(
                f"{self.base_url}/predict_motor",
                json=reading,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 0)
                status = "HEALTHY" if prediction == 1 else "FAULT DETECTED"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.motor_id} - "
                      f"Temp: {reading['temp']}°C, Vib: {reading['vibration']}mm/s, "
                      f"Volt: {reading['voltage']}V, Noise: {reading['noise']}dB - {status}")
                
                return result
            else:
                print(f"Error sending data: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None
    
    def simulate_fault_condition(self):
        """Simulate various fault conditions"""
        fault_types = [
            "high_temp", "high_vibration", "voltage_fluctuation", 
            "excessive_noise", "multiple_faults"
        ]
        
        if random.random() < self.fault_probability:
            self.current_fault = random.choice(fault_types)
            self.fault_duration = random.randint(3, 8)  # 3-8 readings
            print(f"\n⚠️  FAULT SIMULATION: {self.current_fault.upper()} detected!")
    
    def run_simulation(self, interval=2):
        """Run continuous IoT simulation"""
        print(f"🚀 Starting IoT Motor Simulator for {self.motor_id}")
        print(f"📡 Sending data every {interval} seconds to {self.base_url}")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        reading_count = 0
        
        try:
            while self.running:
                reading_count += 1
                
                # Simulate fault conditions
                if self.current_fault and self.fault_duration > 0:
                    reading = self.generate_fault_reading(self.current_fault)
                    self.fault_duration -= 1
                    if self.fault_duration == 0:
                        print("✅ Fault condition resolved")
                        self.current_fault = None
                else:
                    reading = self.generate_normal_reading()
                    self.simulate_fault_condition()
                
                # Add motor ID and timestamp
                reading['motor_id'] = self.motor_id
                reading['timestamp'] = datetime.now().isoformat()
                reading['reading_id'] = reading_count
                
                # Send to Flask app
                self.send_reading(reading)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping IoT simulator for {self.motor_id}")
            self.running = False

def main():
    simulator = MotorIOTSimulator()
    
    print("Motor IoT Device Simulator")
    print("=" * 40)
    print("This simulates a real IoT device continuously monitoring motor health")
    print("The device sends sensor data every 2 seconds to your Flask app")
    print("Fault conditions are randomly simulated for testing")
    print("=" * 40)
    
    try:
        simulator.run_simulation(interval=2)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()






