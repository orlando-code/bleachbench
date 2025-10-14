from datetime import datetime
import xarray as xa


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


def get_bounds(ds: xa.Dataset, var_key: str) -> tuple[float, float]:
    """Get the minimum and maximum values of a variable in a dataset."""
    min_val, max_val = ds[var_key].values.min(), ds[var_key].values.max()
    return min_val, max_val


def regress_climatology(clim_for_month: xa.DataArray) -> tuple[xa.DataArray, xa.DataArray]:
    """
    Accepting a dataarray for a single month, calculate the linear regression
    coefficients (slope and intercept) of the monthly average temperatures over 
    the 28-year period (1985-2012)
    
    Args:
        clim_for_month (xa.DataArray): Dataarray for a single month

    Returns:
        Tuple of slope (m) and intercept (c), so one can compute y = m*x + c.
    """
    x = clim_for_month.time.dt.year.values
    y = clim_for_month
    # calculate values for broadcasting
    x_mean = x.mean()
    y_mean = y.mean(dim='time')
    x_da = xa.DataArray(x, dims=['time'], coords={'time': clim_for_month['time']})
    cov_xy = ((x_da - x_mean) * (y - y_mean)).mean(dim='time')
    var_x = ((x_da - x_mean) ** 2).mean(dim='time')
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict_climatology(slope: xa.DataArray, intercept: xa.DataArray, x_target: float) -> xa.DataArray:
    """
    Predict climatology at a given x_target (year) given linear slope and intercept.
    y = slope * x_target + intercept
    
    Args:
        slope (xa.DataArray): Slope of the linear regression
        intercept (xa.DataArray): Intercept of the linear regression
        x_target (float): Year to predict the climatology for

    Returns:
        xa.DataArray: Predicted climatology at the given x_target
    """
    return intercept + slope * x_target
        