import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_3d_trajectory(
    ax,
    trajectory,
    color='blue',
    min_width=0.5,
    max_width=4,
    label=None,
):
    """
    Plot a 3D trajectory with line width increasing over time.
    """

    trajectory = np.asarray(trajectory)

    if trajectory.shape[0] < 2:
        return

    # Normalize time from 0 -> 1
    t = np.linspace(0, 1, trajectory.shape[0])

    widths = min_width + (max_width - min_width) * t

    for i in range(len(trajectory) - 1):

        ax.plot(
            trajectory[i:i+2, 0],
            trajectory[i:i+2, 1],
            trajectory[i:i+2, 2],
            color=color,
            linewidth=widths[i],
            label=label if i == 0 else None
        )


def plot_beatwise_share(
        trajectories,
        response_times,
        beat_times,
        colors,
        Y=None,
        binsize=0.001,
        n_cols=4,
        n_phase_points=100,
        N=None,
        plot=True,
        save_path=None,
):
    """
    Plot every beat as a separate panel for:

        1. PLS1 vs PLS2 vs PLS3 (3D)
        2. PLS1 vs PLS2       (2D)
        3. PLS1 vs PLS3       (2D)
        4. PLS2 vs PLS3       (2D)

    Each beat is plotted against normalized beat phase:
        0 = beat onset
        1 = next beat

    Different stimulus conditions are overlaid within each panel.

    All 2D subplots within a figure share x and y axes.
    """

    if set(trajectories) != set(beat_times):
        raise ValueError(
            "trajectories and beat_times must contain the same keys"
        )

    # Number of beats in the longest sequence
    n_beats = max(
        len(bt)
        for bt in beat_times.values()
    )

    n_rows = int(np.ceil(n_beats / n_cols))

    # Common normalized phase axis
    phase = np.linspace(0, 1, n_phase_points)

    # ---------------------------------------------------------
    # Create the four figures
    # ---------------------------------------------------------
    if plot:
        fig_3d = plt.figure(
            figsize=(4 * n_cols, 4 * n_rows)
        )
        axes_3d = []

        fig_12, axes_12 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False
        )

        fig_13, axes_13 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False
        )

        fig_23, axes_23 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False
        )

    # ---------------------------------------------------------
    # Calculate global axis limits across ALL trajectories
    # ---------------------------------------------------------
    all_data = np.concatenate(
        [
            np.asarray(traj)[:, :3]
            for traj in trajectories.values()
        ],
        axis=0
    )

    pls1_lim = (all_data[:, 0].min(), all_data[:, 0].max())
    pls2_lim = (all_data[:, 1].min(), all_data[:, 1].max())
    pls3_lim = (all_data[:, 2].min(), all_data[:, 2].max())

    output_traj = {}
    output_time = {}
    for beat_idx in range(n_beats):
        if beat_idx not in output_traj.keys():
            output_traj[beat_idx] = {}
            output_time[beat_idx] = {}
        # -----------------------------------------------------
        # Get axes for this beat
        # -----------------------------------------------------
        if plot:
            ax_3d = fig_3d.add_subplot(
                n_rows,
                n_cols,
                beat_idx + 1,
                projection='3d'
            )

            ax_3d.set_xlim(pls1_lim)
            ax_3d.set_ylim(pls2_lim)
            ax_3d.set_zlim(pls3_lim)

            row = beat_idx // n_cols
            col = beat_idx % n_cols

            ax_12 = axes_12[row, col]
            ax_13 = axes_13[row, col]
            ax_23 = axes_23[row, col]

        # -----------------------------------------------------
        # Plot each stimulus condition
        # -----------------------------------------------------
        for name, traj in trajectories.items():
            bt = np.asarray(beat_times[name])
            t = np.asarray(response_times[name])

            if beat_idx >= len(bt):
                print("  SKIPPED: beat doesn't exist")
                continue

            start = bt[beat_idx]

            if beat_idx + 1 < len(bt):
                end = bt[beat_idx + 1]
                mask = (t >= start) & (t < end)
            else:
                if len(bt) >= 2:
                    end = start + (bt[-1] - bt[-2])
                else:
                    print("  SKIPPED: only one beat")
                    continue
                mask = t >= start

            if mask.sum() < 2:
                print("  SKIPPED: fewer than 2 points")
                continue

            segment = traj[mask, :]
            segment_time = t[mask]


            # Convert time to normalized beat phase
            segment_phase = (
                (segment_time - start)
                / (end - start)
            )

            # Remove duplicate phase values
            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )

            segment = segment[unique_idx]
            segment_phase = unique_phase

            # -------------------------------------------------
            # Interpolate onto common phase axis
            # -------------------------------------------------

            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(segment.shape[1])
            ])
            output_traj[beat_idx][name] = interpolated
            output_time[beat_idx][name] = phase.copy()

            if plot:
                c = colors[name]

                # -------------------------------------------------
                # Linewidth changes continuously over beat phase
                # -------------------------------------------------
                linewidths = np.linspace(0.3, 6.0, len(interpolated))
                linewidths = linewidths ** 1.5
                linewidths = (
                        0.3
                        + (linewidths - linewidths.min())
                        / (linewidths.max() - linewidths.min())
                        * (6.0 - 0.3)
                )

                # Normalize back to desired linewidth range
                linewidths = (
                        0.5
                        + (linewidths - linewidths.min())
                        / (linewidths.max() - linewidths.min())
                        * (5.0 - 0.5)
                )

                # -------------------------------------------------
                # 3D
                # -------------------------------------------------

                for i in range(len(interpolated) - 1):
                    ax_3d.plot(
                        interpolated[i:i + 2, 0],
                        interpolated[i:i + 2, 1],
                        interpolated[i:i + 2, 2],
                        color=c,
                        linewidth=linewidths[i],
                    )
                    axes_3d.append(ax_3d)

                # -------------------------------------------------
                # PLS1 vs PLS2
                # -------------------------------------------------

                for i in range(len(interpolated) - 1):
                    ax_12.plot(
                        interpolated[i:i + 2, 0],
                        interpolated[i:i + 2, 1],
                        color=c,
                        linewidth=linewidths[i],
                    )

                # -------------------------------------------------
                # PLS1 vs PLS3
                # -------------------------------------------------

                for i in range(len(interpolated) - 1):
                    ax_13.plot(
                        interpolated[i:i + 2, 0],
                        interpolated[i:i + 2, 2],
                        color=c,
                        linewidth=linewidths[i],
                    )

                # -------------------------------------------------
                # PLS2 vs PLS3
                # -------------------------------------------------

                for i in range(len(interpolated) - 1):
                    ax_23.plot(
                        interpolated[i:i + 2, 1],
                        interpolated[i:i + 2, 2],
                        color=c,
                        linewidth=linewidths[i],
                    )
        if plot:
            # -----------------------------------------------------
            # Format 3D axes
            # -----------------------------------------------------

            ax_3d.set_title(f'Beat {beat_idx + 1}')

            ax_3d.set_xlabel('PLS1')
            ax_3d.set_ylabel('PLS2')
            ax_3d.set_zlabel('PLS3')

            # -----------------------------------------------------
            # Format 2D axes
            # -----------------------------------------------------

            ax_12.set_title(f'Beat {beat_idx + 1}')
            ax_12.set_xlabel('PLS1')
            ax_12.set_ylabel('PLS2')
            ax_12.grid(alpha=0.2)

            ax_13.set_title(f'Beat {beat_idx + 1}')
            ax_13.set_xlabel('PLS1')
            ax_13.set_ylabel('PLS3')
            ax_13.grid(alpha=0.2)

            ax_23.set_title(f'Beat {beat_idx + 1}')
            ax_23.set_xlabel('PLS2')
            ax_23.set_ylabel('PLS3')
            ax_23.grid(alpha=0.2)

    if plot:
        # ---------------------------------------------------------
        # Hide unused panels
        # ---------------------------------------------------------

        for beat_idx in range(n_beats, n_rows * n_cols):

            row = beat_idx // n_cols
            col = beat_idx % n_cols

            axes_12[row, col].axis('off')
            axes_13[row, col].axis('off')
            axes_23[row, col].axis('off')

        # ---------------------------------------------------------
        # Make the 2D axes explicitly use identical limits
        # ---------------------------------------------------------
        #
        # sharex/sharey already synchronizes them, but this makes
        # the behavior explicit and robust when some panels are
        # empty or have no data.
        # ---------------------------------------------------------

        for axes in (axes_12, axes_13, axes_23):

            # Get limits from the first populated axis
            xlims = []
            ylims = []

            for row in axes:
                for ax in row:
                    if ax.has_data():
                        xlims.append(ax.get_xlim())
                        ylims.append(ax.get_ylim())

            if xlims:
                xmin = min(x[0] for x in xlims)
                xmax = max(x[1] for x in xlims)

                ymin = min(y[0] for y in ylims)
                ymax = max(y[1] for y in ylims)

                for row in axes:
                    for ax in row:
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)

        # ---------------------------------------------------------
        # Common titles
        # ---------------------------------------------------------
        fig_3d.suptitle(f'N={N} neurons', fontsize=15)
        fig_12.suptitle(f'N={N} neurons', fontsize=15)
        fig_13.suptitle(f'N={N} neurons', fontsize=15)
        fig_23.suptitle(f'N={N} neurons', fontsize=15)

        # ---------------------------------------------------------
        # Tight layout
        # ---------------------------------------------------------

        fig_3d.tight_layout()
        fig_12.tight_layout()
        fig_13.tight_layout()
        fig_23.tight_layout()

        # ---------------------------------------------------------
        # Save or show figures
        # ---------------------------------------------------------

        if save_path is not None:
            import os
            os.makedirs(save_path, exist_ok=True)

            fig_3d.savefig(
                os.path.join(save_path, "beatwise_3D.svg"),
                bbox_inches="tight"
            )

            fig_12.savefig(
                os.path.join(save_path, "beatwise_PLS1_PLS2.svg"),
                bbox_inches="tight"
            )

            fig_13.savefig(
                os.path.join(save_path, "beatwise_PLS1_PLS3.svg"),
                bbox_inches="tight"
            )

            fig_23.savefig(
                os.path.join(save_path, "beatwise_PLS2_PLS3.svg"),
                bbox_inches="tight"
            )

            plt.close(fig_3d)
            plt.close(fig_12)
            plt.close(fig_13)
            plt.close(fig_23)

        else:
            plt.show()

    return output_traj, output_time

def plot_beatwise(
        trajectories,
        response_times,
        beat_times,
        colors,
        Y=None,
        binsize=0.001,
        n_cols=4,
        n_phase_points=100,
        N=None
):
    """
    Plot every beat as a separate panel for:

        1. PLS1 vs PLS2 vs PLS3 (3D)
        2. PLS1 vs PLS2       (2D)
        3. PLS1 vs PLS3       (2D)
        4. PLS2 vs PLS3       (2D)

    Each beat is plotted against normalized beat phase:

        0 = beat onset
        1 = next beat

    Different stimulus conditions are overlaid within each panel.

    Unlike the previous version, every 2D subplot has completely
    independent x and y axes.
    """

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    if set(trajectories) != set(beat_times):
        raise ValueError(
            "trajectories and beat_times must contain the same keys"
        )

    if set(trajectories) != set(response_times):
        raise ValueError(
            "trajectories and response_times must contain the same keys"
        )

    for name in trajectories:

        if len(trajectories[name]) != len(response_times[name]):
            raise ValueError(
                f"{name}: trajectory and response_times have different "
                f"lengths ({len(trajectories[name])} vs "
                f"{len(response_times[name])})"
            )

    # ---------------------------------------------------------
    # Number of beats
    # ---------------------------------------------------------

    n_beats = max(
        len(bt)
        for bt in beat_times.values()
    )

    n_rows = int(np.ceil(n_beats / n_cols))

    # Common normalized phase axis
    phase = np.linspace(0, 1, n_phase_points)

    # ---------------------------------------------------------
    # Create figures
    # ---------------------------------------------------------

    fig_3d = plt.figure(
        figsize=(4 * n_cols, 4 * n_rows)
    )

    fig_12, axes_12 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    fig_13, axes_13 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    fig_23, axes_23 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    # ---------------------------------------------------------
    # Global 3D limits
    #
    # Keep the 3D plots on the same scale so that trajectories
    # can be compared across beats.
    # ---------------------------------------------------------

    all_data = np.concatenate(
        [
            np.asarray(traj)[:, :3]
            for traj in trajectories.values()
        ],
        axis=0
    )

    if not np.isfinite(all_data).all():
        raise ValueError(
            "trajectories contain NaN or infinite values"
        )

    pls1_lim = (
        all_data[:, 0].min(),
        all_data[:, 0].max()
    )

    pls2_lim = (
        all_data[:, 1].min(),
        all_data[:, 1].max()
    )

    pls3_lim = (
        all_data[:, 2].min(),
        all_data[:, 2].max()
    )

    # Add a small margin so trajectories aren't sitting directly
    # on the edge of the axes.
    def add_margin(lim, fraction=0.05):

        low, high = lim
        span = high - low

        if span == 0:
            span = 1.0

        margin = span * fraction

        return (
            low - margin,
            high + margin
        )

    pls1_lim = add_margin(pls1_lim)
    pls2_lim = add_margin(pls2_lim)
    pls3_lim = add_margin(pls3_lim)

    # ---------------------------------------------------------
    # Loop through beats
    # ---------------------------------------------------------

    for beat_idx in range(n_beats):

        # -----------------------------------------------------
        # 3D axis
        # -----------------------------------------------------

        ax_3d = fig_3d.add_subplot(
            n_rows,
            n_cols,
            beat_idx + 1,
            projection="3d"
        )

        ax_3d.set_xlim(pls1_lim)
        ax_3d.set_ylim(pls2_lim)
        ax_3d.set_zlim(pls3_lim)

        # -----------------------------------------------------
        # 2D axes
        # -----------------------------------------------------

        row = beat_idx // n_cols
        col = beat_idx % n_cols

        ax_12 = axes_12[row, col]
        ax_13 = axes_13[row, col]
        ax_23 = axes_23[row, col]

        # -----------------------------------------------------
        # Plot each stimulus condition
        # -----------------------------------------------------

        for name, traj in trajectories.items():

            bt = np.asarray(beat_times[name])
            t = np.asarray(response_times[name])
            traj = np.asarray(traj)

            # -------------------------------------------------
            # Does this beat exist?
            # -------------------------------------------------

            if beat_idx >= len(bt):
                continue

            start = bt[beat_idx]

            # -------------------------------------------------
            # Determine end of beat
            # -------------------------------------------------

            if beat_idx + 1 < len(bt):

                end = bt[beat_idx + 1]

                mask = (
                    (t >= start) &
                    (t < end)
                )

            else:

                # Last beat
                if len(bt) >= 2:

                    end = start + (
                        bt[-1] - bt[-2]
                    )

                else:
                    continue

                mask = t >= start

            # -------------------------------------------------
            # Need at least two points
            # -------------------------------------------------

            if mask.sum() < 2:
                continue

            segment = traj[mask, :3]
            segment_time = t[mask]

            # -------------------------------------------------
            # Normalize time to beat phase
            # -------------------------------------------------

            beat_duration = end - start

            if beat_duration <= 0:
                continue

            segment_phase = (
                (segment_time - start)
                / beat_duration
            )

            # -------------------------------------------------
            # Sort and remove duplicate phase values
            # -------------------------------------------------

            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )

            segment = segment[unique_idx]
            segment_phase = unique_phase

            if len(segment_phase) < 2:
                continue

            # -------------------------------------------------
            # Interpolate onto common phase axis
            # -------------------------------------------------

            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(3)
            ])

            # -------------------------------------------------
            # Line width changes over phase
            # -------------------------------------------------

            linewidths = np.linspace(
                0.3,
                6.0,
                len(interpolated)
            )

            linewidths = linewidths ** 1.5

            if linewidths.max() != linewidths.min():

                linewidths = (
                    0.3
                    + (
                        (linewidths - linewidths.min())
                        / (
                            linewidths.max()
                            - linewidths.min()
                        )
                    )
                    * (6.0 - 0.3)
                )

            linewidths = (
                0.5
                + (
                    (linewidths - linewidths.min())
                    / (
                        linewidths.max()
                        - linewidths.min()
                    )
                )
                * (5.0 - 0.5)
            )

            c = colors[name]

            # -------------------------------------------------
            # 3D trajectory
            # -------------------------------------------------

            for i in range(len(interpolated) - 1):

                ax_3d.plot(
                    interpolated[i:i + 2, 0],
                    interpolated[i:i + 2, 1],
                    interpolated[i:i + 2, 2],
                    color=c,
                    linewidth=linewidths[i]
                )

            # -------------------------------------------------
            # PLS1 vs PLS2
            # -------------------------------------------------

            for i in range(len(interpolated) - 1):

                ax_12.plot(
                    interpolated[i:i + 2, 0],
                    interpolated[i:i + 2, 1],
                    color=c,
                    linewidth=linewidths[i]
                )

            # -------------------------------------------------
            # PLS1 vs PLS3
            # -------------------------------------------------

            for i in range(len(interpolated) - 1):

                ax_13.plot(
                    interpolated[i:i + 2, 0],
                    interpolated[i:i + 2, 2],
                    color=c,
                    linewidth=linewidths[i]
                )

            # -------------------------------------------------
            # PLS2 vs PLS3
            # -------------------------------------------------

            for i in range(len(interpolated) - 1):

                ax_23.plot(
                    interpolated[i:i + 2, 1],
                    interpolated[i:i + 2, 2],
                    color=c,
                    linewidth=linewidths[i]
                )

        # -----------------------------------------------------
        # Format 3D
        # -----------------------------------------------------

        ax_3d.set_title(
            f"Beat {beat_idx + 1}"
        )

        ax_3d.set_xlabel("PLS1")
        ax_3d.set_ylabel("PLS2")
        ax_3d.set_zlabel("PLS3")

        # -----------------------------------------------------
        # Format 2D
        # -----------------------------------------------------

        ax_12.set_title(
            f"Beat {beat_idx + 1}"
        )

        ax_12.set_xlabel("PLS1")
        ax_12.set_ylabel("PLS2")
        ax_12.grid(alpha=0.2)

        ax_13.set_title(
            f"Beat {beat_idx + 1}"
        )

        ax_13.set_xlabel("PLS1")
        ax_13.set_ylabel("PLS3")
        ax_13.grid(alpha=0.2)

        ax_23.set_title(
            f"Beat {beat_idx + 1}"
        )

        ax_23.set_xlabel("PLS2")
        ax_23.set_ylabel("PLS3")
        ax_23.grid(alpha=0.2)

    # ---------------------------------------------------------
    # Hide unused panels
    # ---------------------------------------------------------

    for beat_idx in range(
        n_beats,
        n_rows * n_cols
    ):

        row = beat_idx // n_cols
        col = beat_idx % n_cols

        axes_12[row, col].axis("off")
        axes_13[row, col].axis("off")
        axes_23[row, col].axis("off")

    # ---------------------------------------------------------
    # Titles
    # ---------------------------------------------------------

    fig_3d.suptitle(
        f"N={N} neurons",
        fontsize=15
    )

    fig_12.suptitle(
        f"N={N} neurons",
        fontsize=15
    )

    fig_13.suptitle(
        f"N={N} neurons",
        fontsize=15
    )

    fig_23.suptitle(
        f"N={N} neurons",
        fontsize=15
    )

    # ---------------------------------------------------------
    # Layout
    #
    # Don't use tight_layout on the 2D figures.
    # ---------------------------------------------------------

    fig_3d.tight_layout()

    fig_12.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.06,
        top=0.94,
        wspace=0.30,
        hspace=0.30
    )

    fig_13.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.06,
        top=0.94,
        wspace=0.30,
        hspace=0.30
    )

    fig_23.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.06,
        top=0.94,
        wspace=0.30,
        hspace=0.30
    )

    # ---------------------------------------------------------
    # Show
    # ---------------------------------------------------------

    plt.show()

def plot_color_legend(colors):
    """
    Create a standalone legend showing the color coding for
    tempo and Regular vs. Irregular conditions.
    """
    from matplotlib.patches import Patch

    tempos = []
    for color in colors.keys():
        tempos.append(int(color.split(' ')[0]))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    handles = []

    for tempo in tempos:
        regular = f'{tempo} Regular'
        irregular = f'{tempo} Irregular'

        if f'{tempo} Regular' in colors.keys():
            handles.append(
                Patch(
                    facecolor=colors[regular],
                    label=f'{tempo} Regular'
                )
            )
        if f'{tempo} Irregular' in colors.keys():
            handles.append(
                Patch(
                    facecolor=colors[irregular],
                    label=f'{tempo} Irregular'
                )
            )

    ax.legend(
        handles=handles,
        loc='center',
        ncol=2,
        frameon=False,
        fontsize=12
    )

    ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_trajectory_correlation_heatmap(
        trajectories,
        response_times,
        beat_times,
        dims=3,
        n_beats=None,
        n_phase_points=100,
        figsize=(7, 6),
        cmap='RdBu_r'
):
    """
    Calculate pairwise trajectory-shape correlations within each beat
    and plot the mean correlation across beats as a heatmap.

    Correlation is calculated separately for each beat, using normalized
    beat phase, then averaged across beats.

    Parameters
    ----------
    trajectories : dict
        {name: trajectory_array}
        Arrays should have shape (time, dimensions).

    response_times : dict
        {name: time_array}

    beat_times : dict
        {name: beat_time_array}

    dims : int
        Number of trajectory dimensions to use.

    n_beats : int or None
        Number of beat windows to include.

    n_phase_points : int
        Number of points used to represent each beat.

    figsize : tuple
        Figure size.

    cmap : str
        Colormap for correlation heatmap.

    Returns
    -------
    mean_corr : pandas.DataFrame
        Mean pairwise trajectory correlation across beats.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    names = list(trajectories.keys())

    if set(names) != set(response_times.keys()):
        raise ValueError(
            "trajectories and response_times must have the same keys"
        )

    if set(names) != set(beat_times.keys()):
        raise ValueError(
            "trajectories and beat_times must have the same keys"
        )

    if n_beats is None:
        n_beats = min(
            len(beat_times[name]) - 1
            for name in names
        )

    phase = np.linspace(0, 1, n_phase_points)

    # Store correlations for every beat
    beat_correlations = []

    # ---------------------------------------------------------
    # Loop over beats
    # ---------------------------------------------------------

    for beat_idx in range(n_beats):

        beat_trajectories = {}

        for name in names:

            traj = np.asarray(trajectories[name])
            t = np.asarray(response_times[name])
            bt = np.asarray(beat_times[name])

            if beat_idx + 1 >= len(bt):
                continue

            start = bt[beat_idx]
            end = bt[beat_idx + 1]

            mask = (
                (t >= start) &
                (t < end)
            )

            if mask.sum() < 2:
                continue

            segment = traj[mask, :dims]
            segment_time = t[mask]

            # Convert to normalized beat phase
            segment_phase = (
                (segment_time - start)
                / (end - start)
            )

            # Remove duplicate phase values
            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )

            segment = segment[unique_idx]
            segment_phase = unique_phase

            if len(segment_phase) < 2:
                continue

            # Interpolate trajectory onto common phase axis
            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(dims)
            ])

            beat_trajectories[name] = interpolated

        # -----------------------------------------------------
        # Correlation between trajectory SHAPES
        # -----------------------------------------------------

        beat_corr = pd.DataFrame(
            np.eye(len(names)),
            index=names,
            columns=names
        )

        for i, name1 in enumerate(names):

            for j, name2 in enumerate(names):

                if j <= i:
                    continue

                if (
                    name1 not in beat_trajectories or
                    name2 not in beat_trajectories
                ):
                    continue

                traj1 = beat_trajectories[name1]
                traj2 = beat_trajectories[name2]

                # Flatten dimensions × time
                x = traj1.flatten()
                y = traj2.flatten()

                if np.std(x) == 0 or np.std(y) == 0:
                    corr = np.nan
                else:
                    corr = np.corrcoef(x, y)[0, 1]

                beat_corr.loc[name1, name2] = corr
                beat_corr.loc[name2, name1] = corr

        beat_correlations.append(beat_corr)

    # ---------------------------------------------------------
    # Average correlation across beats
    # ---------------------------------------------------------

    corr_stack = np.stack([
        x.values.astype(float)
        for x in beat_correlations
    ])

    mean_corr = pd.DataFrame(
        np.nanmean(corr_stack, axis=0),
        index=names,
        columns=names
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        mean_corr.values,
        vmin=-1,
        vmax=1,
        cmap=cmap
    )

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    ax.set_xticklabels(names, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)

    ax.set_title(
        'Mean trajectory-shape correlation across beats'
    )

    # Add correlation values
    for i in range(len(names)):
        for j in range(len(names)):

            value = mean_corr.iloc[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f'{value:.2f}',
                    ha='center',
                    va='center'
                )

    fig.colorbar(
        im,
        ax=ax,
        label='Pearson correlation'
    )

    fig.tight_layout()
    plt.show()

    return mean_corr

def plot_beat_by_beat_correlation_heatmap(
        trajectories,
        response_times,
        beat_times,
        dims=3,
        n_phase_points=100,
        n_beats=None,
        cmap='RdBu_r'
):
    names = list(trajectories.keys())
    if set(names) != set(response_times.keys()):
        raise ValueError(
            "trajectories and response_times must have the same keys"
        )

    if set(names) != set(beat_times.keys()):
        raise ValueError(
            "trajectories and beat_times must have the same keys"
        )
    if n_beats is None:
        n_beats = min(
            len(beat_times[name]) - 1
            for name in names
        )
    phase = np.linspace(0, 1, n_phase_points)
    beat_correlations = []
    for beat_idx in range(n_beats):
        beat_trajectories = {}
        for name in names:
            traj = np.asarray(trajectories[name])
            t = np.asarray(response_times[name])
            bt = np.asarray(beat_times[name])

            if beat_idx + 1 >= len(bt):
                continue

            start = bt[beat_idx]
            end = bt[beat_idx + 1]
            mask = (t>=start) & (t<end)
            if mask.sum() < 2:
                continue
            segment = traj[mask, :dims]
            segment_time = t[mask]
            segment_phase = (
                    (segment_time - start)
                    / (end - start)
            )
            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )
            segment = segment[unique_idx]
            segment_phase = unique_phase
            if len(segment_phase) < 2:
                continue
            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(dims)
            ])
            beat_trajectories[name] = interpolated

            import pandas as pd
            beat_corr = pd.DataFrame(
                np.eye(len(names)),
                index=names,
                columns=names
            )
            for i, name1 in enumerate(names):

                for j, name2 in enumerate(names):

                    if j <= i:
                        continue

                    if (
                            name1 not in beat_trajectories or
                            name2 not in beat_trajectories
                    ):
                        continue

                    traj1 = beat_trajectories[name1]
                    traj2 = beat_trajectories[name2]

                    # Flatten dimensions × time
                    x = traj1.flatten()
                    y = traj2.flatten()

                    if np.std(x) == 0 or np.std(y) == 0:
                        corr = np.nan
                    else:
                        corr = np.corrcoef(x, y)[0, 1]

                    beat_corr.loc[name1, name2] = corr
                    beat_corr.loc[name2, name1] = corr

        beat_correlations.append(beat_corr)

    cols = 4
    nrows = int(np.ceil(len(beat_correlations) / cols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=cols,
        figsize=(4 * cols, 4 * nrows)
    )

    for idx, (beat, ax) in enumerate(zip(beat_correlations, axes.flat)):
        im = ax.imshow(
            beat.values,
            vmin=-1,
            vmax=1,
            cmap=cmap
        )
        ax.set_title(f'Beat {idx + 1}')
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))

        ax.set_xticklabels(names, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        # Add correlation values
        for i in range(len(names)):
            for j in range(len(names)):

                value = beat.iloc[i, j]

                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f'{value:.2f}',
                        ha='center',
                        va='center'
                    )

    plt.tight_layout()


from pathlib import Path
def save_beatwise(
        trajectories,
        response_times,
        beat_times,
        colors,
        save_dir=None,
        n_phase_points=100,
        figsize=(8, 8),
        dpi=300,
        formats=('png',),
        linewidth_min=0.75,
        linewidth_max=4.0,
        close_figures=True
):
    """
    Save each beat/projection as an individual figure.

    Linewidth increases over normalized beat phase:

        phase = 0  --> linewidth_min
        phase = 1  --> linewidth_max

    Creates:
        - PLS1 vs PLS2
        - PLS1 vs PLS3
        - PLS2 vs PLS3
        - PLS1 vs PLS2 vs PLS3

    All figures within a projection use identical axis limits.

    No titles or legends are added.
    """

    # =========================================================
    # Validate
    # =========================================================

    if set(trajectories) != set(beat_times):
        raise ValueError(
            "trajectories and beat_times must contain the same keys"
        )

    if set(trajectories) != set(response_times):
        raise ValueError(
            "trajectories and response_times must contain the same keys"
        )

    # =========================================================
    # Save directory
    # =========================================================

    if save_dir is None:
        save_dir = Path.home() / "Downloads" / "PLS_beatwise"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    directories = {
        'PLS1_PLS2': save_dir / 'PLS1_vs_PLS2',
        'PLS1_PLS3': save_dir / 'PLS1_vs_PLS3',
        'PLS2_PLS3': save_dir / 'PLS2_vs_PLS3',
        'PLS1_PLS2_PLS3': save_dir / 'PLS1_PLS2_PLS3',
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Number of beats
    # =========================================================

    n_beats = max(
        len(bt)
        for bt in beat_times.values()
    )

    phase = np.linspace(
        0,
        1,
        n_phase_points
    )

    # =========================================================
    # Extract/interpolate beat trajectories
    # =========================================================

    beat_data = {}

    for beat_idx in range(n_beats):

        beat_data[beat_idx] = {}

        for name, traj in trajectories.items():

            bt = np.asarray(beat_times[name])
            t = np.asarray(response_times[name])
            traj = np.asarray(traj)

            if beat_idx >= len(bt):
                continue

            start = bt[beat_idx]

            # -------------------------------------------------
            # Normal beat
            # -------------------------------------------------

            if beat_idx + 1 < len(bt):

                end = bt[beat_idx + 1]

                mask = (
                    (t >= start) &
                    (t < end)
                )

            # -------------------------------------------------
            # Final beat
            # -------------------------------------------------

            else:

                mask = t >= start

                if len(bt) < 2:
                    continue

                end = start + (bt[-1] - bt[-2])

            if mask.sum() < 2:
                continue

            segment = traj[mask, :3]
            segment_time = t[mask]

            # -------------------------------------------------
            # Normalize time to beat phase
            # -------------------------------------------------

            segment_phase = (
                (segment_time - start)
                / (end - start)
            )

            valid = (
                np.isfinite(segment_phase) &
                (segment_phase >= 0) &
                (segment_phase <= 1)
            )

            segment_phase = segment_phase[valid]
            segment = segment[valid]

            if len(segment_phase) < 2:
                continue

            # Remove duplicate phase values
            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )

            segment = segment[unique_idx]
            segment_phase = unique_phase

            if len(segment_phase) < 2:
                continue

            # -------------------------------------------------
            # Interpolate trajectory
            # -------------------------------------------------

            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(3)
            ])

            beat_data[beat_idx][name] = interpolated

    # =========================================================
    # GLOBAL AXIS LIMITS
    # =========================================================

    all_data = np.concatenate([
        data
        for beat in beat_data.values()
        for data in beat.values()
    ])

    def padded_limits(vmin, vmax, fraction=0.05):

        span = vmax - vmin

        if span == 0:
            span = 1

        padding = fraction * span

        return (
            vmin - padding,
            vmax + padding
        )

    pls1_lim = padded_limits(
        np.min(all_data[:, 0]),
        np.max(all_data[:, 0])
    )

    pls2_lim = padded_limits(
        np.min(all_data[:, 1]),
        np.max(all_data[:, 1])
    )

    pls3_lim = padded_limits(
        np.min(all_data[:, 2]),
        np.max(all_data[:, 2])
    )

    # =========================================================
    # Helper for 2D variable-width trajectories
    # =========================================================

    def plot_variable_width_2d(
            ax,
            x,
            y,
            color
    ):
        """
        Plot a trajectory with linewidth increasing over time.
        """

        points = np.column_stack([
            x,
            y
        ])

        segments = np.stack([
            points[:-1],
            points[1:]
        ], axis=1)

        # Midpoint phase of each segment
        segment_phase = (
            np.arange(len(segments)) + 0.5
        ) / len(segments)

        linewidths = (
            linewidth_min
            + segment_phase
            * (linewidth_max - linewidth_min)
        )

        collection = LineCollection(
            segments,
            linewidths=linewidths,
            colors=color
        )

        ax.add_collection(collection)

    # =========================================================
    # Helper for 3D variable-width trajectories
    # =========================================================

    def plot_variable_width_3d(
            ax,
            x,
            y,
            z,
            color
    ):
        """
        Plot a 3D trajectory with linewidth increasing over time.
        """

        points = np.column_stack([
            x,
            y,
            z
        ])

        segments = np.stack([
            points[:-1],
            points[1:]
        ], axis=1)

        segment_phase = (
            np.arange(len(segments)) + 0.5
        ) / len(segments)

        linewidths = (
            linewidth_min
            + segment_phase
            * (linewidth_max - linewidth_min)
        )

        collection = Line3DCollection(
            segments,
            linewidths=linewidths,
            colors=color
        )

        ax.add_collection3d(collection)

    # =========================================================
    # Generate figures
    # =========================================================

    for beat_idx in range(n_beats):

        data_for_beat = beat_data[beat_idx]

        if not data_for_beat:
            continue

        beat_number = beat_idx + 1

        # =====================================================
        # PLS1 vs PLS2
        # =====================================================

        fig, ax = plt.subplots(
            figsize=figsize
        )

        for name, data in data_for_beat.items():

            plot_variable_width_2d(
                ax,
                data[:, 0],
                data[:, 1],
                colors[name]
            )

        ax.set_xlim(pls1_lim)
        ax.set_ylim(pls2_lim)

        ax.set_xlabel('PLS1')
        ax.set_ylabel('PLS2')

        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()

        for fmt in formats:
            fig.savefig(
                directories['PLS1_PLS2']
                / f'beat_{beat_number:02d}.{fmt}',
                dpi=dpi,
                bbox_inches='tight'
            )

        if close_figures:
            plt.close(fig)

        # =====================================================
        # PLS1 vs PLS3
        # =====================================================

        fig, ax = plt.subplots(
            figsize=figsize
        )

        for name, data in data_for_beat.items():

            plot_variable_width_2d(
                ax,
                data[:, 0],
                data[:, 2],
                colors[name]
            )

        ax.set_xlim(pls1_lim)
        ax.set_ylim(pls3_lim)

        ax.set_xlabel('PLS1')
        ax.set_ylabel('PLS3')

        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()

        for fmt in formats:
            fig.savefig(
                directories['PLS1_PLS3']
                / f'beat_{beat_number:02d}.{fmt}',
                dpi=dpi,
                bbox_inches='tight'
            )

        if close_figures:
            plt.close(fig)

        # =====================================================
        # PLS2 vs PLS3
        # =====================================================

        fig, ax = plt.subplots(
            figsize=figsize
        )

        for name, data in data_for_beat.items():

            plot_variable_width_2d(
                ax,
                data[:, 1],
                data[:, 2],
                colors[name]
            )

        ax.set_xlim(pls2_lim)
        ax.set_ylim(pls3_lim)

        ax.set_xlabel('PLS2')
        ax.set_ylabel('PLS3')

        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()

        for fmt in formats:
            fig.savefig(
                directories['PLS2_PLS3']
                / f'beat_{beat_number:02d}.{fmt}',
                dpi=dpi,
                bbox_inches='tight'
            )

        if close_figures:
            plt.close(fig)

        # =====================================================
        # 3D
        # =====================================================

        fig = plt.figure(
            figsize=figsize
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        for name, data in data_for_beat.items():

            plot_variable_width_3d(
                ax,
                data[:, 0],
                data[:, 1],
                data[:, 2],
                colors[name]
            )

        ax.set_xlim(pls1_lim)
        ax.set_ylim(pls2_lim)
        ax.set_zlim(pls3_lim)

        ax.set_xlabel('PLS1')
        ax.set_ylabel('PLS2')
        ax.set_zlabel('PLS3')

        fig.tight_layout()

        for fmt in formats:
            fig.savefig(
                directories['PLS1_PLS2_PLS3']
                / f'beat_{beat_number:02d}.{fmt}',
                dpi=dpi,
                bbox_inches='tight'
            )

        if close_figures:
            plt.close(fig)

    print(
        f"Saved figures to:\n{save_dir}"
    )

    print("\nGlobal axis limits:")
    print(f"PLS1: {pls1_lim}")
    print(f"PLS2: {pls2_lim}")
    print(f"PLS3: {pls3_lim}")

def plot_beatwise_old(
    trajectories,
    response_times,
    beat_times,
    colors,
    Y=None,
    binsize=0.001,
    n_cols=4,
    n_phase_points=100
):
    """
    Plot every beat as a separate 3D panel.

    Each beat is plotted against normalized beat phase:
        0   = beat onset
        1   = next beat

    Different stimulus conditions are overlaid within each panel.
    """

    if set(trajectories) != set(beat_times):
        raise ValueError(
            "trajectories and beat_times must contain the same keys"
        )

    # Number of beats in each sequence
    n_beats = max(
        len(bt) - 1
        for bt in beat_times.values()
    )

    n_rows = int(np.ceil(n_beats / n_cols))

    fig = plt.figure(
        figsize=(4 * n_cols, 4 * n_rows)
    )

    # Common normalized phase axis
    phase = np.linspace(0, 1, n_phase_points)

    for beat_idx in range(n_beats):

        ax = fig.add_subplot(
            n_rows,
            n_cols,
            beat_idx + 1,
            projection='3d'
        )

        for name, traj in trajectories.items():

            bt = np.asarray(beat_times[name])

            # Sequence doesn't contain this beat
            if beat_idx >= len(bt) - 1:
                continue

            # Time boundaries of this beat
            start = bt[beat_idx]
            end = bt[beat_idx + 1]

            # Time axis for trajectory
            t = response_times[name]

            # Select this beat
            mask = (t >= start) & (t < end)

            if mask.sum() < 2:
                continue

            segment = traj[mask, :3]
            segment_time = t[mask]

            # Convert time to normalized beat phase
            segment_phase = (
                (segment_time - start) /
                (end - start)
            )

            # Remove duplicate phase values if necessary
            unique_phase, unique_idx = np.unique(
                segment_phase,
                return_index=True
            )

            segment = segment[unique_idx]
            segment_phase = unique_phase

            # Interpolate each PC onto common phase axis
            interpolated = np.column_stack([
                np.interp(
                    phase,
                    segment_phase,
                    segment[:, dim]
                )
                for dim in range(3)
            ])

            # Plot the normalized trajectory
            ax.plot(
                interpolated[:, 0],
                interpolated[:, 1],
                interpolated[:, 2],
                color=colors[name],
                label=name
            )

        ax.set_title(f'Beat {beat_idx + 1}')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    # Put one legend on the figure rather than every subplot
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels)
        )

    if Y is not None:
        fig, axes = plt.subplots(len(Y.keys()), 1, figsize=(4*n_rows, 10), sharex=True)

        for l, ax in zip(Y.keys(), axes):
            t = np.arange(len(Y[l])) / 1000
            ax.plot(t, Y[l], color=colors[l])

            for b in beat_times[l]:
                ax.axvline(b, color='black', linestyle='--', alpha=0.3)
            ax.set_title(l)
    plt.tight_layout()
    plt.show()

def plot_pls_vs_time(
    trajectories,
    response_times,
    beat_times=None,
    colors=None,
    Y=None
):
    """
    Plot each PLS dimension against real time.

    Creates:
        1. PLS1 vs time
        2. PLS2 vs time
        3. PLS3 vs time

    Different stimulus conditions are overlaid.

    Parameters
    ----------
    trajectories : dict
        Dictionary of PLS trajectories.
        Each array should have shape (n_time_points, >=3).

    response_times : dict
        Dictionary of time vectors corresponding to each trajectory.

    beat_times : dict, optional
        Dictionary of beat times for each condition.

    colors : dict, optional
        Dictionary mapping condition names to colors.

    Y : dict, optional
        Optional stimulus waveform dictionary to plot separately.
    """

    if colors is None:
        colors = {}

    # ---------------------------------------------------------
    # Check inputs
    # ---------------------------------------------------------

    if set(trajectories) != set(response_times):
        raise ValueError(
            "trajectories and response_times must contain the same keys"
        )

    # ---------------------------------------------------------
    # Create three figures
    # ---------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig3, ax3 = plt.subplots(figsize=(10, 5))

    axes = {
        0: ax1,
        1: ax2,
        2: ax3
    }

    # ---------------------------------------------------------
    # Plot each condition
    # ---------------------------------------------------------

    for name, traj in trajectories.items():

        traj = np.asarray(traj)
        t = np.asarray(response_times[name])

        if traj.ndim != 2 or traj.shape[1] < 3:
            raise ValueError(
                f"{name}: trajectory must have shape "
                f"(n_time_points, >=3)"
            )

        if len(t) != len(traj):
            raise ValueError(
                f"{name}: response_times has length {len(t)}, "
                f"trajectory has length {len(traj)}"
            )

        c = colors.get(name, None)

        # PLS1
        ax1.plot(
            t,
            traj[:, 0],
            color=c,
            label=name
        )

        # PLS2
        ax2.plot(
            t,
            traj[:, 1],
            color=c,
            label=name
        )

        # PLS3
        ax3.plot(
            t,
            traj[:, 2],
            color=c,
            label=name
        )

    # ---------------------------------------------------------
    # Add beat markers
    # ---------------------------------------------------------

    if beat_times is not None:

        for name, bt in beat_times.items():

            c = colors.get(name, None)

            for b in np.asarray(bt):

                for ax in (ax1, ax2, ax3):
                    ax.axvline(
                        b,
                        color=c,
                        linestyle='--',
                        alpha=0.25
                    )

    # ---------------------------------------------------------
    # Format axes
    # ---------------------------------------------------------

    ax1.set_title('PLS1 vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('PLS1')
    ax1.grid(alpha=0.2)
    ax1.legend()

    ax2.set_title('PLS2 vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('PLS2')
    ax2.grid(alpha=0.2)
    ax2.legend()

    ax3.set_title('PLS3 vs Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('PLS3')
    ax3.grid(alpha=0.2)
    ax3.legend()

    # ---------------------------------------------------------
    # Optional stimulus plots
    # ---------------------------------------------------------

    if Y is not None:

        fig_y, axes_y = plt.subplots(
            len(Y),
            1,
            figsize=(10, 3 * len(Y)),
            sharex=True,
            squeeze=False
        )

        axes_y = axes_y[:, 0]

        for ax, name in zip(axes_y, Y.keys()):

            y = np.asarray(Y[name])
            t = np.arange(len(y)) / 1000

            ax.plot(
                t,
                y,
                color=colors.get(name, None)
            )

            if beat_times is not None and name in beat_times:

                for b in beat_times[name]:
                    ax.axvline(
                        b,
                        color='black',
                        linestyle='--',
                        alpha=0.3
                    )

            ax.set_title(name)
            ax.set_ylabel('Y')

        axes_y[-1].set_xlabel('Time (s)')
        fig_y.tight_layout()

    # ---------------------------------------------------------
    # Layout and display
    # ---------------------------------------------------------

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    plt.show()