#!/usr/bin/env python3
"""
Quick test to verify the motor shutdown fix with the exact values from the user's report
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_exact_values():
    """Test with the exact values that caused the false shutdown"""
    
    print("🔧 Testing Motor Shutdown Fix")
    print("=" * 40)
    
    # Exact values from user's report that caused false shutdown
    test_data = {
        "temp": 30,
        "vibration": 2,
        "voltage": 220,
        "noise": 50
    }
    
    print(f"📊 Testing with sensor values:")
    print(f"   Temperature: {test_data['temp']}°C")
    print(f"   Vibration: {test_data['vibration']}mm/s")
    print(f"   Voltage: {test_data['voltage']}V")
    print(f"   Noise: {test_data['noise']}dB")
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_motor",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            shutdown_triggered = data.get('shutdown_triggered', False)
            detailed_conditions = data.get('detailed_conditions', [])
            
            print(f"🔄 Shutdown triggered: {shutdown_triggered}")
            
            if shutdown_triggered:
                print(f"❌ PROBLEM: Motor shutdown with normal values!")
                print(f"🚨 Shutdown reason: {data.get('shutdown_reason', 'Unknown')}")
                if detailed_conditions:
                    print(f"📊 Detailed conditions:")
                    for condition in detailed_conditions:
                        print(f"   • {condition}")
            else:
                print(f"✅ CORRECT: Motor continues running with normal values")
                print(f"📊 Prediction: {'Healthy' if data.get('prediction') == 1 else 'Fault predicted'}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure your Flask app is running on http://127.0.0.1:5000")

if __name__ == "__main__":
    test_exact_values()
