# general
from datetime import datetime
from pathlib import Path

import requests
import xarray as xa
from dateutil.relativedelta import relativedelta

# custom
# from src.config import config


def download_erdapp_thredds_nc_files(
    var: str = "CRW_SST",
    date_range: tuple[str, str] = ("1985-04-01T12:00:00Z", "2024-04-28T12:00:00Z"),
    lats: tuple[float, float] = (-32, 0),
    lons: tuple[float, float] = (130, 170),
    output_dir=None,
):
    """Download variable data from ERDDAP server for a given date range and spatial extent.

    ERDDAP link: https://pae-paha.pacioos.hawaii.edu/thredds/satellite.html?dataset=dhw_5km

    Parameters
    var (str): The variable to download. Default is "CRW_SST".
    date_range (tuple): A tuple of strings representing the start and end dates in the format "YYYY-MM-DDTHH:MM:SSZ".
    lats (tuple): A tuple of floats representing the minimum and maximum latitudes to download.
    lons (tuple): A tuple of floats representing the minimum and maximum longitudes to download.
    output_dir (Path): The directory to save the downloaded files to.

    var options:
    N.B. "var" value is just captilised portion (e.g. enter "CRW_SST" for "CRW_SST (Celsius)")
        CRW_SST (Celsius) = nighttime sea surface temperature = sea_surface_temperature
        CRW_BAA (1) = coral bleaching alert area = coral_bleaching_alert_area
        CRW_BAA_7D_MAX (1) = coral bleaching alert area 7-day maximum = coral_bleaching_alert_area_7day_maximum
        CRW_DHW (Celsius weeks) = coral bleaching degree heating week = coral_bleaching_degree_heating_week
        CRW_HOTSPOT (Celsius) = coral bleaching hot spots = coral_bleaching_hot_spots
        CRW_SSTANOMALY (Celsius) = nighttime sea surface temperature anomaly = sea_surface_temperature_anomaly
    """
    # set correct base URL for the variable
    url_base = f"https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.nc?{var}%5B"
    # set spatial constraints
    url_end = (
        f"%5D%5B({min(lats)}):1:({max(lats)})%5D%5B({min(lons)}):1:({max(lons)})%5D"
    )

    if not output_dir:  # if no output directory path provided
        output_dir = Path(config.sst_dir) / var
        output_dir.mkdir(parents=True, exist_ok=True)

    # if date_range is a tuple of strings, convert start and end dates to datetime objects
    if isinstance(date_range[0], str):
        dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ") for date in date_range]
    elif isinstance(date_range[0], datetime):
        dates = date_range

    # assign start_date to oldest date
    start_date = min(dates)
    end_date = max(dates)

    print(
        f"Downloading {var} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} to {output_dir}\n"
    )
    # Iterate over month-by-month chunks
    while start_date < end_date:
        # Calculate the end of the month
        next_month = start_date + relativedelta(months=1)
        end_of_month = next_month.replace(day=1) - relativedelta(days=1)

        # Construct the URL for the current month
        month_url = f"{url_base}({start_date.isoformat()}Z):1:({end_of_month.isoformat()}Z){url_end}"

        output_fp = output_dir / f"{var}_{start_date.strftime('%Y-%m')}.nc"
        if not output_fp.exists():
            # Download the data
            response = requests.get(month_url)
            if response.status_code == 200:  # successful request
                data = xa.open_dataset(response.content)
                data.to_netcdf(output_fp)
                print(
                    f"Downloaded data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} "
                    f"and saved to {output_fp}"
                )
            elif response.status_code == 404:
                print(
                    f"Data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} not found ({response.status_code}) at {response.url}.\nAre you sure it exists?"
                )
            else:
                print(
                    f"Failed to download data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')}.\nError: \n{request.text}"
                )
        else:
            print(
                f"\tData for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} already exists. Skipping download..."
            )
        # iterate to next month
        start_date = next_month

    print("\nDownload complete.")

