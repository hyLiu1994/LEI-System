from pydantic import BaseModel, Field, field_validator
from typing import Any

ALLOWED_COLUMNS = [
    '# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
    'Navigational status', 'ROT', 'SOG', 'COG', 'Heading', 'IMO', 'Callsign',
    'Name', 'Ship type', 'Cargo type', 'Width', 'Length',
    'Type of position fixing device', 'Draught', 'Destination', 'ETA',
    'Data source type'
]


class DeleteGPSPointByMMIDTool(BaseModel):
    """Delete a GPS points record by MMID."""
    mmsi: int = Field(..., description="The MMID of the GPS points to delete.")


class UpdateGPSPointColByIndexTool(BaseModel):
    """Update a specific column value for a GPS point by its index.
    or update Latitude value 57.00 index=1
    """
    index: int = Field(..., description="The index of the GPS point to update.")
    column_name: str = Field(..., description="The name of the column to update. Must be one of the allowed columns.")
    value: Any = Field(..., description="The new value to set for the specified column.")

    @field_validator('column_name')
    def validate_column_name(cls, v):
        if v not in ALLOWED_COLUMNS:
            raise ValueError(f"Invalid column name. Allowed columns are: {', '.join(ALLOWED_COLUMNS)}")
        return v
