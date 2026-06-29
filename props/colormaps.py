def get_colormap(cmap_name, fallback_name):
    from matplotlib import colormaps

    try:
        return colormaps[cmap_name]
    except KeyError:
        if cmap_name in ("NWSRef", "NWSVel", "NWS_SPW", "NWS_CC"):
            import cmweather  # noqa: F401

            return colormaps[cmap_name]
        return colormaps[fallback_name]
