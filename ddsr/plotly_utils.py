_DEFAULT_AXIS_VISIBLE_RADIUS_M = 110
_DEFAULT_AXIS_BOTTOM_PLANE_M = -1
_DEFAULT_AXIS_HEIGHT_M = 10

def setup_layout(
    fig,
    bev_radius_m=_DEFAULT_AXIS_VISIBLE_RADIUS_M,
    bottom_plane_m=_DEFAULT_AXIS_BOTTOM_PLANE_M,
    height_m=_DEFAULT_AXIS_HEIGHT_M,
):
    """
    Configures default figure layout for plotly in 3D. Points outside of these bounds will not be shown.
    Call this function right before showing the figure to ensure it isn't overwritten.

    :param fig: A plotly figure to build on top of.
    :param bev_radius: Birds-Eye View (BEV) radius in meters.
    :param bottom_plane_m: Where the bottom plane is defined in the figure in meters.
    :param height_m: Height of our viewing figure in meters.
    """
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=30, range=[-bev_radius_m, bev_radius_m]),
            yaxis=dict(nticks=30, range=[-bev_radius_m, bev_radius_m]),
            zaxis=dict(nticks=30, range=[bottom_plane_m, bottom_plane_m + height_m]),
            # The goal of this aspect ratio is to keep everything 1:1:1 given the cropped x, y, z axis.
            aspectratio=dict(x=(2 * bev_radius_m) / height_m, y=(2 * bev_radius_m) / height_m, z=1),
        )
    )
