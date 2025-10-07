import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def format_geo_axes(
    ax: plt.Axes, extent: tuple | list = (-180, 180, -40, 50)
) -> plt.Axes:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, alpha=0.3, zorder=-10)
    ax.add_feature(cfeature.COASTLINE, edgecolor="lightgray", zorder=-1)
    ax.add_feature(
        cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.1, zorder=-1
    )

    return ax
