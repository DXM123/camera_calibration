import dataclasses
from typing import Optional

import yaml


@dataclasses.dataclass
class DeWarpConfig:
    # Config Default Variables - Enter their values according to your Checkerboard, normal 64 (8x8) -1 inner corners only
    no_of_columns: int  # number of columns of your Checkerboard
    no_of_rows: int  # number of rows of your Checkerboard
    square_size: float  # size of square on the Checkerboard in mm -> TODO: This is no longer required?
    min_cap: int  # minimum or images to be collected by capturing (Default is 10), minimum is 3

    # Assuming the soccer field is 22 x 14 meters - old
    soccer_field_width: float
    soccer_field_length: float

    # Field Size and other dimensions for MSL field defaults see `falcon_config.yaml`
    field_length: float  # meters
    field_width: float  # meters
    penalty_area_length: float  # E, meters
    penalty_area_width: float  # C, meters
    goal_area_length: float  # F, meters
    goal_area_width: float  # D, meters
    center_circle_radius: float  # H, meters
    spot_radius: float
    goal_depth: float  # Goal depth,
    goal_width: float  # Goal width 2m for this field -> 2.4m allowed?
    line_width: float  # K, meters
    ppm: int  # pixels per meter
    safe_zone: float  # Safety zone around the field, meters

    ### Total Field Size
    field_length_total: float = dataclasses.field(
        init=False
    )  # field_length + 2 * safe_zone  -- Adding safety zone to the length
    field_width_total: float = dataclasses.field(
        init=False
    )  # field_width + 2 * safe_zone  -- Adding safety zone to the width

    def __post_init__(self):
        self.field_length_total = self.field_length + 2 * self.safe_zone
        self.field_width_total = self.field_width + 2 * self.safe_zone


def get_config(path_to_config: Optional[str] = None) -> DeWarpConfig:
    """Takes a str to a `.yaml` or if `None` loads the default Falcon config."""
    if not path_to_config:
        path_to_config = "falcon_config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    return DeWarpConfig(**config)
