from pydantic import BaseModel, Field, field_validator
from typing import List

ALLOWED_COLUMNS = [
    '# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
    'Navigational status', 'ROT', 'SOG', 'COG', 'Heading', 'IMO', 'Callsign',
    'Name', 'Ship type', 'Cargo type', 'Width', 'Length',
    'Type of position fixing device', 'Draught', 'Destination', 'ETA',
    'Data source type'
]


class StatisticCategoryByScTool(BaseModel):
    """Statistic different vessel categories by special column and Select the top_n items to draw the image ."""
    top_n: int = Field(..., description="top_n items to draw the image.")
    column_name: str = Field(...,
                             description="The name of the column to statistic or classify or group. Must be one of the allowed columns.")

    @field_validator('column_name')
    def validate_column_name(cls, v):
        if v not in ALLOWED_COLUMNS:
            raise ValueError(f"Invalid column name. Allowed columns are: {', '.join(ALLOWED_COLUMNS)}")
        return v


class GpsPositionsByTrajectoryIdsTool(BaseModel):
    """Get GPS positions by a list of trajectory sequence nums.
       or Get Gps trajectory bs a list of trajectory sequence nums.
    """
    trajectory_sequence_nums: List[str] = Field(...,
                                                description="A list of trajectory sequence nums to fetch the GPS positions.")
