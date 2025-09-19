#!/usr/bin/env python3
"""
Motor Shutdown Test Script
This script simulates critical motor conditions to test automatic shutdown functionality.
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:5000"

def test_motor_shutdown():
    """Test automatic motor shutdown with critical conditions"""
    
    print("🚀 Testing Motor Automatic Shutdown System")
    print("=" * 50)
    
    # Test cases for critical shutdown conditions
    test_cases = [
        {
            "name": "Extreme Overheating",
            "data": {"temp": 95, "vibration": 2.0, "voltage": 230, "noise": 75},
            "expected_shutdown": True
        },
        {
            "name": "Extreme Vibration", 
            "data": {"temp": 60, "vibration": 9.0, "voltage": 230, "noise": 75},
            "expected_shutdown": True
        },
        {
            "name": "Extreme Voltage",
            "data": {"temp": 60, "vibration": 2.0, "voltage": 170, "noise": 75},
            "expected_shutdown": True
        },
        {
            "name": "Extreme Noise",
            "data": {"temp": 60, "vibration": 2.0, "voltage": 230, "noise": 105},
            "expected_shutdown": True
        },
        {
            "name": "Multiple Critical Conditions",
            "data": {"temp": 87, "vibration": 7.0, "voltage": 185, "noise": 98},
            "expected_shutdown": True
        },
        {
            "name": "High Temperature (Not Critical)",
            "data": {"temp": 85, "vibration": 2.0, "voltage": 230, "noise": 75},
            "expected_shutdown": False
        },
        {
            "name": "High Vibration (Not Critical)",
            "data": {"temp": 60, "vibration": 7.0, "voltage": 230, "noise": 75},
            "expected_shutdown": False
        },
        {
            "name": "Normal Operation",
            "data": {"temp": 55, "vibration": 2.5, "voltage": 230, "noise": 75},
            "expected_shutdown": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"   Sensor Data: {test_case['data']}")
        
        try:
            # Send prediction request
            response = requests.post(
                f"{BASE_URL}/predict_motor",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                shutdown_triggered = data.get('shutdown_triggered', False)
                shutdown_reason = data.get('shutdown_reason', 'None')
                detailed_conditions = data.get('detailed_conditions', [])
                
                print(f"   ✅ Request successful")
                print(f"   🔄 Shutdown triggered: {shutdown_triggered}")
                
                if shutdown_triggered:
                    print(f"   🚨 Shutdown reason: {shutdown_reason}")
                    print(f"   ⚠️  Critical conditions: {data.get('critical_conditions', [])}")
                    if detailed_conditions:
                        print(f"   📊 Detailed analysis:")
                        for condition in detailed_conditions:
                            print(f"      • {condition}")
                else:
                    print(f"   ✅ Motor continues running normally")
                
                # Check if result matches expectation
                if shutdown_triggered == test_case['expected_shutdown']:
                    print(f"   ✅ Test PASSED - Expected: {test_case['expected_shutdown']}, Got: {shutdown_triggered}")
                else:
                    print(f"   ❌ Test FAILED - Expected: {test_case['expected_shutdown']}, Got: {shutdown_triggered}")
                
                # Check motor status
                status_response = requests.get(f"{BASE_URL}/motor/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   📊 Motor Status: {'RUNNING' if status_data['is_running'] else 'SHUTDOWN'}")
                
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")
        
        # Wait between tests
        time.sleep(2)
    
    print(f"\n🏁 Testing Complete!")
    print(f"   Check the web interface at {BASE_URL} to see the motor status panel")

def test_motor_control():
    """Test manual motor control (start/stop)"""
    
    print(f"\n🎮 Testing Motor Control")
    print("=" * 30)
    
    try:
        # Test stop motor
        print("🛑 Testing motor stop...")
        response = requests.post(
            f"{BASE_URL}/motor/control",
            json={"action": "stop"},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stop result: {data['message']}")
            print(f"   📊 Motor running: {data['motor_state']['is_running']}")
        
        time.sleep(2)
        
        # Test start motor
        print("▶️ Testing motor start...")
        response = requests.post(
            f"{BASE_URL}/motor/control",
            json={"action": "start"},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Start result: {data['message']}")
            print(f"   📊 Motor running: {data['motor_state']['is_running']}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Control test error: {e}")

if __name__ == "__main__":
    print("🔧 Motor Shutdown System Test")
    print("Make sure your Flask app is running on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop testing\n")
    
    try:
        test_motor_shutdown()
        test_motor_control()
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
