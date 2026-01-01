# Dynamic Assessment Count Feature - Bug Fix

## Issue
When using custom subject configurations, the system threw an error:
```
'<' not supported between instances of 'NoneType' and 'int'
```

## Root Cause
In `subject_config.py`, the `fill_missing_with_average()` function was checking if a field exists in the dictionary (`if field in obtained_marks`) but not checking if the value was `None`. 

When a field existed but had a `None` value, it would be added to `complete_data` as `None`, and later validation would try to compare `None < 0`, causing the error.

## Fix Applied
Changed the logic in `fill_missing_with_average()` from:
```python
if field in obtained_marks:
    complete_data[field] = obtained_marks[field]
```

To:
```python
value = obtained_marks.get(field)
if value is not None:
    complete_data[field] = value
```

This explicitly checks that the value is not `None` before using it, preventing the comparison error.

## Testing
The fix ensures that:
1. Only non-None values are used from obtained_marks
2. Missing or None fields are filled with the student's average
3. Attendance defaults to 100.0 if missing or None
4. All 10 required fields are guaranteed to have valid numeric values

## Status
✅ **FIXED** - The None comparison error should no longer occur when using custom subject configurations.
