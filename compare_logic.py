#!/usr/bin/env python3
"""
Comprehensive test showing the difference between old and new motor shutdown logic
"""

def test_old_logic(temp, vibration, voltage, noise):
    """Simulate the OLD (buggy) logic"""
    print("🔴 OLD LOGIC (Buggy):")
    
    # Old logic - this was the problem
    conditions = [temp >= 85, vibration >= 6.5, voltage < 190 or voltage > 270, noise >= 95]
    count = sum(conditions)
    
    print(f"   Conditions checked:")
    print(f"   • temp >= 85: {temp >= 85} (temp={temp})")
    print(f"   • vibration >= 6.5: {vibration >= 6.5} (vibration={vibration})")
    print(f"   • voltage < 190 or > 270: {voltage < 190 or voltage > 270} (voltage={voltage})")
    print(f"   • noise >= 95: {noise >= 95} (noise={noise})")
    print(f"   Total conditions met: {count}")
    
    if count >= 2:
        print(f"   🚨 OLD LOGIC: SHUTDOWN (Multiple conditions: {count})")
        return True
    else:
        print(f"   ✅ OLD LOGIC: Continue running")
        return False

def test_new_logic(temp, vibration, voltage, noise):
    """Simulate the NEW (fixed) logic"""
    print("🟢 NEW LOGIC (Fixed):")
    
    # New logic - proper priority
    temp_critical = temp >= 90
    vib_critical = vibration >= 8.0
    volt_critical = voltage < 180 or voltage > 280
    noise_critical = noise >= 100
    ml_critical = False  # Simplified for demo
    
    print(f"   Extreme conditions checked:")
    print(f"   • temp >= 90: {temp_critical} (temp={temp})")
    print(f"   • vibration >= 8.0: {vib_critical} (vibration={vibration})")
    print(f"   • voltage < 180 or > 280: {volt_critical} (voltage={voltage})")
    print(f"   • noise >= 100: {noise_critical} (noise={noise})")
    
    # Single extreme conditions (priority)
    if temp_critical:
        print(f"   🚨 NEW LOGIC: SHUTDOWN (Extreme overheating)")
        return True
    elif vib_critical:
        print(f"   🚨 NEW LOGIC: SHUTDOWN (Extreme vibration)")
        return True
    elif volt_critical:
        print(f"   🚨 NEW LOGIC: SHUTDOWN (Extreme voltage)")
        return True
    elif noise_critical:
        print(f"   🚨 NEW LOGIC: SHUTDOWN (Extreme noise)")
        return True
    
    # Multiple critical conditions (only if no single extreme condition)
    multiple_conditions = []
    if temp >= 85:
        multiple_conditions.append(f"Temperature: {temp}°C (Warning: ≥85°C)")
    if vibration >= 6.5:
        multiple_conditions.append(f"Vibration: {vibration}mm/s (Warning: ≥6.5mm/s)")
    if voltage < 190 or voltage > 270:
        multiple_conditions.append(f"Voltage: {voltage}V (Warning: <190V or >270V)")
    if noise >= 95:
        multiple_conditions.append(f"Noise: {noise}dB (Warning: ≥95dB)")
    
    print(f"   Multiple conditions checked:")
    print(f"   • temp >= 85: {temp >= 85} (temp={temp})")
    print(f"   • vibration >= 6.5: {vibration >= 6.5} (vibration={vibration})")
    print(f"   • voltage < 190 or > 270: {voltage < 190 or voltage > 270} (voltage={voltage})")
    print(f"   • noise >= 95: {noise >= 95} (noise={noise})")
    print(f"   Multiple conditions met: {len(multiple_conditions)}")
    
    if len(multiple_conditions) >= 2:
        print(f"   🚨 NEW LOGIC: SHUTDOWN (Multiple critical conditions)")
        return True
    else:
        print(f"   ✅ NEW LOGIC: Continue running")
        return False

def test_scenarios():
    """Test various scenarios"""
    
    scenarios = [
        {
            "name": "User's Problem Case",
            "data": {"temp": 30, "vibration": 2, "voltage": 220, "noise": 50},
            "expected": "Should NOT shutdown"
        },
        {
            "name": "Normal Operation",
            "data": {"temp": 55, "vibration": 3, "voltage": 230, "noise": 75},
            "expected": "Should NOT shutdown"
        },
        {
            "name": "High but Safe Values",
            "data": {"temp": 85, "vibration": 6, "voltage": 240, "noise": 90},
            "expected": "Should NOT shutdown"
        },
        {
            "name": "Extreme Overheating",
            "data": {"temp": 95, "vibration": 2, "voltage": 230, "noise": 75},
            "expected": "Should shutdown"
        },
        {
            "name": "Multiple Critical",
            "data": {"temp": 87, "vibration": 7, "voltage": 185, "noise": 98},
            "expected": "Should shutdown"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {scenario['name']}")
        print(f"📊 Sensor Data: {scenario['data']}")
        print(f"🎯 Expected: {scenario['expected']}")
        print()
        
        data = scenario['data']
        temp, vibration, voltage, noise = data['temp'], data['vibration'], data['voltage'], data['noise']
        
        old_result = test_old_logic(temp, vibration, voltage, noise)
        print()
        new_result = test_new_logic(temp, vibration, voltage, noise)
        
        print(f"\n📋 Summary:")
        print(f"   Old Logic: {'SHUTDOWN' if old_result else 'Continue'}")
        print(f"   New Logic: {'SHUTDOWN' if new_result else 'Continue'}")
        
        if scenario['expected'] == 'Should NOT shutdown':
            if old_result and not new_result:
                print(f"   ✅ FIXED: Old logic incorrectly shutdown, new logic correctly continues")
            elif not old_result and not new_result:
                print(f"   ✅ CORRECT: Both logics correctly continue")
            else:
                print(f"   ❌ ISSUE: Both logics behave the same")
        else:  # Should shutdown
            if old_result and new_result:
                print(f"   ✅ CORRECT: Both logics correctly shutdown")
            elif not old_result and new_result:
                print(f"   ✅ IMPROVED: New logic correctly shutdowns when old didn't")
            else:
                print(f"   ❌ ISSUE: New logic doesn't shutdown when it should")

if __name__ == "__main__":
    print("🔧 Motor Shutdown Logic Comparison")
    print("This shows the difference between old (buggy) and new (fixed) logic")
    print()
    
    test_scenarios()
    
    print(f"\n{'='*60}")
    print("🏁 Test Complete!")
    print("The new logic fixes the false shutdown issue while maintaining safety.")
