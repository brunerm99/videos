def cfar(
    x: np.ndarray,
    num_ref_cells: int,
    num_gap_cells: int,
    bias: float = 1,
    method=np.mean,
):
    pad = int((num_ref_cells + num_gap_cells))
    window_mean = np.pad(                                                                   # Pad front/back since n_windows < n_points
        method(                                                                             # Apply input method to remaining compute cells
            np.delete(                                                                      # Remove guard cells, CUT from computation
                sliding_window_view(x, (num_ref_cells * 2) + (num_gap_cells * 2)),          # Windows of x including CUT, guard cells, and compute cells
                np.arange(int(num_ref_cells), num_ref_cells + (num_gap_cells * 2) + 1),     # Get indices of guard cells, CUT
                axis=1), 
            axis=1
        ), (pad - 1, pad),                                                               
        constant_values=(np.nan, np.nan)                                                    # Fill with NaNs
    ) * bias                                                                                # Multiply output by bias over which cell is not noise
    return window_mean