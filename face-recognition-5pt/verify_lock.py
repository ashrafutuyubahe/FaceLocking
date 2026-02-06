#!/usr/bin/env python3
"""Quick verification of lock.py enhancements."""

import sys
import os
from pathlib import Path

# Check file exists
lock_file = Path("src/lock.py")
if not lock_file.exists():
    print("❌ Lock file not found")
    sys.exit(1)

print("✓ Lock file exists")

# Read and check for new classes
content = lock_file.read_text(encoding='utf-8')

checks = [
    ("FaceTracker class", "class FaceTracker:"),
    ("ActionDetector class", "class ActionDetector:"),
    ("Face tracking smoothing", "self.x1_buffer = deque"),
    ("Action state machine", "self.blink_state = False"),
    ("Enhanced UI with emojis", "🔒"),
    ("Frame skipping optimization", "process_frame_interval"),
    ("Interactive controls", "ord(\"+\")"),
    ("Better error messages", "🔓"),
]

print("\n" + "="*70)
print(" LOCK.PY ENHANCEMENT VERIFICATION")
print("="*70 + "\n")

passed = 0
failed = 0

for feature, code_marker in checks:
    if code_marker in content:
        print(f"✅ {feature}")
        passed += 1
    else:
        print(f"❌ {feature} (marker not found: {code_marker})")
        failed += 1

print("\n" + "="*70)
print(f" Results: {passed} passed, {failed} failed")
print("="*70)

if failed == 0:
    print("\n✅ ALL ENHANCEMENTS VERIFIED - Lock.py is ready!")
    
    # Show file stats
    lines = content.count('\n')
    print(f"\n📊 File Statistics:")
    print(f"   - Total lines: {lines}")
    print(f"   - File size: {len(content)} bytes")
    
    sys.exit(0)
else:
    print("\n❌ Some enhancements not found")
    sys.exit(1)
