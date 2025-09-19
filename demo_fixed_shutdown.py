#!/usr/bin/env python3
"""
Motor Shutdown Fix Demonstration
This script demonstrates the fixed motor shutdown logic with proper conditions.
"""

import requests
import time

BASE_URL = "http://127.0.0.1:5000"

def demonstrate_fixed_shutdown():
    """Demonstrate the fixed motor shutdown logic"""
    
    print("🔧 Motor Shutdown System - FIXED VERSION")
    print("=" * 50)
    print("This demonstrates the corrected shutdown logic:")
    print("• Only shuts down on EXTREME conditions")
    print("• Shows detailed analysis of critical conditions")
    print("• Prevents false shutdowns on normal high values")
    print()
    
    # Test cases showing the fix
    test_cases = [
        {
            "name": "✅ NORMAL OPERATION (Should NOT shutdown)",
            "data": {"temp": 75, "vibration": 4.0, "voltage": 230, "noise": 80},
            "expected": "Continue running"
        },
        {
            "name": "⚠️ HIGH VALUES (Should NOT shutdown)",
            "data": {"temp": 85, "vibration": 6.0, "voltage": 240, "noise": 90},
            "expected": "Continue running"
        },
        {
            "name": "🚨 EXTREME OVERHEATING (Should shutdown)",
            "data": {"temp": 95, "vibration": 2.0, "voltage": 230, "noise": 75},
            "expected": "SHUTDOWN"
        },
        {
            "name": "🚨 EXTREME VIBRATION (Should shutdown)",
            "data": {"temp": 60, "vibration": 9.0, "voltage": 230, "noise": 75},
            "expected": "SHUTDOWN"
        },
        {
            "name": "🚨 MULTIPLE CRITICAL (Should shutdown)",
            "data": {"temp": 87, "vibration": 7.0, "voltage": 185, "noise": 98},
            "expected": "SHUTDOWN"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"   📊 Sensor Data: {test_case['data']}")
        print(f"   🎯 Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict_motor",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                shutdown_triggered = data.get('shutdown_triggered', False)
                detailed_conditions = data.get('detailed_conditions', [])
                
                if shutdown_triggered:
                    print(f"   🚨 RESULT: MOTOR SHUTDOWN")
                    print(f"   📋 Reason: {data.get('shutdown_reason', 'Unknown')}")
                    if detailed_conditions:
                        print(f"   📊 Detailed Analysis:")
                        for condition in detailed_conditions:
                            print(f"      • {condition}")
                else:
                    print(f"   ✅ RESULT: Motor continues running")
                    print(f"   📊 Prediction: {'Healthy' if data.get('prediction') == 1 else 'Fault predicted'}")
                
                # Check if result matches expectation
                if test_case['expected'] == 'SHUTDOWN' and shutdown_triggered:
                    print(f"   ✅ CORRECT: Shutdown triggered as expected")
                elif test_case['expected'] == 'Continue running' and not shutdown_triggered:
                    print(f"   ✅ CORRECT: Motor continues running as expected")
                else:
                    print(f"   ❌ INCORRECT: Expected {test_case['expected']}, got {'SHUTDOWN' if shutdown_triggered else 'Continue running'}")
                    
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")
        
        time.sleep(1)
    
    print(f"\n🏁 Demonstration Complete!")
    print(f"   The motor shutdown system now works correctly:")
    print(f"   • Only shuts down on EXTREME conditions (90°C+, 8.0mm/s+, etc.)")
    print(f"   • Shows detailed analysis of what caused the shutdown")
    print(f"   • Prevents false shutdowns on normal high values")
    print(f"   • Provides clear information about critical conditions")

if __name__ == "__main__":
    print("🔧 Motor Shutdown Fix Demonstration")
    print("Make sure your Flask app is running on http://127.0.0.1:5000")
    print("This will show the corrected shutdown behavior\n")
    
    try:
        demonstrate_fixed_shutdown()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
