import random
import time
import requests
from datetime import datetime

class WaterIOTSimulator:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.running = False
        self.device_id = "WATER_001"

        # Normal water quality ranges (based on drinking water standards)
        self.ph_range = (6.5, 8.5)  # pH
        self.hardness_range = (50, 150)  # mg/L
        self.sulfate_range = (100, 250)  # mg/L
        self.turbidity_range = (0.1, 1.0)  # NTU

        # Contamination simulation parameters
        self.contamination_probability = 0.05  # 5% chance of contamination per reading
        self.current_contamination = None
        self.contamination_duration = 0

    def generate_normal_reading(self):
        """Generate normal water quality readings"""
        return {
            'ph': round(random.uniform(*self.ph_range), 1),
            'hardness': round(random.uniform(*self.hardness_range), 1),
            'sulfate': round(random.uniform(*self.sulfate_range), 1),
            'turbidity': round(random.uniform(*self.turbidity_range), 1)
        }

    def generate_contaminated_reading(self, contamination_type):
        """Generate readings with specific contamination conditions"""
        base = self.generate_normal_reading()

        if contamination_type == "acidic_ph":
            base['ph'] = round(random.uniform(4.0, 6.0), 1)
        elif contamination_type == "basic_ph":
            base['ph'] = round(random.uniform(9.0, 11.0), 1)
        elif contamination_type == "high_hardness":
            base['hardness'] = round(random.uniform(300, 500), 1)
        elif contamination_type == "high_sulfate":
            base['sulfate'] = round(random.uniform(400, 600), 1)
        elif contamination_type == "high_turbidity":
            base['turbidity'] = round(random.uniform(5.0, 10.0), 1)
        elif contamination_type == "multiple_contaminants":
            base['ph'] = round(random.uniform(5.0, 6.0), 1)
            base['hardness'] = round(random.uniform(250, 400), 1)
            base['sulfate'] = round(random.uniform(350, 500), 1)
            base['turbidity'] = round(random.uniform(3.0, 8.0), 1)

        return base

    def send_reading(self, reading):
        """Send reading to the Flask app"""
        try:
            response = requests.post(
                f"{self.base_url}/predict_water",
                json=reading,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 0)
                status = "POTABLE" if prediction == 1 else "NOT POTABLE"

                print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.device_id} - "
                      f"pH: {reading['ph']}, Hardness: {reading['hardness']}mg/L, "
                      f"Sulfate: {reading['sulfate']}mg/L, Turbidity: {reading['turbidity']}NTU - {status}")

                return result
            else:
                print(f"Error sending data: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None

    def simulate_contamination(self):
        """Simulate various contamination conditions"""
        contamination_types = [
            "acidic_ph", "basic_ph", "high_hardness",
            "high_sulfate", "high_turbidity", "multiple_contaminants"
        ]

        if random.random() < self.contamination_probability:
            self.current_contamination = random.choice(contamination_types)
            self.contamination_duration = random.randint(3, 8)  # 3-8 readings
            print(f"\n⚠️  CONTAMINATION SIMULATION: {self.current_contamination.upper()} detected!")

    def run_simulation(self, interval=3):
        """Run continuous IoT simulation"""
        print(f"🚀 Starting IoT Water Quality Simulator for {self.device_id}")
        print(f"📡 Sending data every {interval} seconds to {self.base_url}")
        print("Press Ctrl+C to stop\n")

        self.running = True
        reading_count = 0

        try:
            while self.running:
                reading_count += 1

                # Simulate contamination conditions
                if self.current_contamination and self.contamination_duration > 0:
                    reading = self.generate_contaminated_reading(self.current_contamination)
                    self.contamination_duration -= 1
                    if self.contamination_duration == 0:
                        print("✅ Contamination condition resolved")
                        self.current_contamination = None
                else:
                    reading = self.generate_normal_reading()
                    self.simulate_contamination()

                # Add device ID and timestamp
                reading['device_id'] = self.device_id
                reading['timestamp'] = datetime.now().isoformat()
                reading['reading_id'] = reading_count

                # Send to Flask app
                self.send_reading(reading)

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping IoT simulator for {self.device_id}")
            self.running = False

def main():
    simulator = WaterIOTSimulator()

    print("Water Quality IoT Device Simulator")
    print("=" * 40)
    print("This simulates a real IoT device continuously monitoring water quality")
    print("The device sends sensor data every 3 seconds to your Flask app")
    print("Contamination conditions are randomly simulated for testing")
    print("=" * 40)

    try:
        simulator.run_simulation(interval=3)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()
