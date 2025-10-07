from datetime import datetime


def parse_date(date):
    """
    Parse a date string or datetime object into a datetime object.
    Accepts:
        - YYYY
        - YYYYMM
        - YYYYMMDD
        - YYYY-MM-DDTHH:MM:SSZ (ISO 8601)
        - datetime object (returns as is)
    """
    if isinstance(date, datetime):
        return date
    if not isinstance(date, str):
        date = str(date)
    date = date.split("T")[0].replace("-", "")
    formats = {4: "%Y", 6: "%Y%m", 8: "%Y%m%d"}
    fmt = formats.get(len(date))
    if fmt:
        return datetime.strptime(date, fmt)
    raise ValueError(f"Invalid date format: {date}")


def get_bounds(ds, var_key):
    min_val, max_val = ds[var_key].values.min(), ds[var_key].values.max()
    return min_val, max_val
