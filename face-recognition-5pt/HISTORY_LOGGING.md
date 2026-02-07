# History Logging Feature

## Overview

The face locking system now automatically saves all lock events and detected actions to history log files.

## What Gets Logged

All actions are automatically saved to timestamped log files in `data/history/`:

### Logged Events:

1. **LOCK_START** - When the system locks onto the target face
2. **LOCK_RELEASE** - When the lock is released (target face lost)
3. **FACE_MOVED_LEFT** - Target face moved to the left
4. **FACE_MOVED_RIGHT** - Target face moved to the right
5. **EYE_BLINK** - Target person blinked
6. **SMILE** - Target person smiled
7. **MULTI_PERSON** - Multiple people detected in frame
8. **SESSION_END** - Face locking session ended

## Log File Format

Each session creates a new history file with format:

```
{identity_name}_history_{YYYYMMDDHHMMSS}.txt
```

Example: `ashrafu_history_20260207123045.txt`

### File Structure:

```
# Face Lock history: ashrafu
# Started: 2026-02-07 12:30:45
# Format: timestamp  action_type  description
# ---

2026-02-07 12:30:45.123  LOCK_START            Locked onto ashrafu
2026-02-07 12:30:48.456  EYE_BLINK             👁 BLINK
2026-02-07 12:30:52.789  SMILE                 😊 SMILE
2026-02-07 12:31:05.234  FACE_MOVED_LEFT       ⬅ LEFT
2026-02-07 12:31:08.567  FACE_MOVED_RIGHT      ➡ RIGHT
2026-02-07 12:31:15.890  MULTI_PERSON          Multiple people: ashrafu, bedo, UNKNOWN
2026-02-07 12:31:20.123  LOCK_RELEASE          Released lock on ashrafu
2026-02-07 12:32:00.456  SESSION_END           Face locking session ended
```

## How to Use

1. **Run the face locking system:**

   ```bash
   python -m src.lock
   ```

2. **Select the identity to lock onto** when prompted

3. **Perform actions** in front of the camera:
   - Move your face left/right
   - Blink your eyes
   - Smile
   - Have multiple people in frame

4. **Check the logs**:
   - All actions are saved in real-time to `data/history/`
   - Each session creates a new timestamped file
   - Files are human-readable text format

## Benefits

- ✅ **Automatic logging** - No manual intervention needed
- ✅ **Complete audit trail** - Every action is timestamped and recorded
- ✅ **Real-time saving** - Actions are written immediately
- ✅ **Session tracking** - Each session gets its own file
- ✅ **Human-readable** - Easy to review and analyze
- ✅ **Multi-person tracking** - Logs when multiple people are detected

## Example Use Cases

1. **Security monitoring** - Review who accessed the system and when
2. **Behavior analysis** - Analyze patterns in detected actions
3. **System debugging** - Track when locks/unlocks occur
4. **Activity logging** - Complete record of all face interactions
5. **Compliance** - Maintain audit trail for regulated environments

## Location

All history files are saved to:

```
data/history/{identity}_history_{timestamp}.txt
```

These files are automatically created and managed by the system.
