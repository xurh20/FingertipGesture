# FingertipGesture

This is a novel project with gesture recognition and prediction of fingertips. Inspired from Sensel Morph and Sensel API, started from <https://github.com/morinted/sensel-api>, this is only for lab research.

## Sensel API

The Sensel API allows users to communicate with Sensel devices.

[Sensel API Primer](http://guide.sensel.com/api/)

[Sensel Lib Documentation](http://guide.sensel.com/sensel_h/)

[Sensel Decompress Lib Documentation](http://guide.sensel.com/sensel_decompress_h/)

## Get Stated

### Examples

```cmd
cd sensel-examples/sensel-python/
python example_4_forces_control.py
```

### FingertipGestureApp

#### Recording

```cmd
python Record.py [-b|f|i|m|p|r|s] [-n SavedDirName]
```

| Arguments | Description |
| --------- | ----------- |
| `-b`, `--bound` | print bound parameters |
| `-f`, `--feedback` | record frames with visual feedback |
| `-i`, `--interactive` | print realtime pressure |
| `-m`, `-max` | print max frames |
| `-n`, `-name` | specify the name of the saving directory |
| `-p`, `--predict` | print predicted results |
| `-r`, `-record` | save frames |
| `-s`, `-sum` | print sum frames |
Or run `python Record.py -h` for help.

When the sensel is ready:

- Press `Enter` to start a record
- When recording, press `Enter` again to stop. The record will be saved as file immediately.
- The record count will automatically increase, every five attempts for one character. The app exits after all over.
- When NOT recording, input `p` to rewrite the previous record.
- Input `q` ANY time to stop recording and exit.
- Other inputs will be ignored.
