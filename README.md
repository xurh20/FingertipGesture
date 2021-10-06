# FingertipGesture

This is a novel project with gesture recognition and prediction of fingertips. Inspired from Sensel Morph and Sensel API, started from https://github.com/morinted/sensel-api, this is only for lab research.

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

```cmd
python Record.py [-b|m|r|s]
```

| Arguments | Description |
| --------- | ----------- |
| -b, --bound | print bound parameters |
| -m, -max | print max frames |
| -r, -record | save frames |
| -s, -sum | print sum frames |
Or run `python Record.py -h` for help.