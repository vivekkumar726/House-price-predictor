from pydantic import BaseModel, Field
from typing import Literal


class HouseRequest(BaseModel):
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=1)
    bathrooms: int = Field(..., ge=1)
    stories: int = Field(..., ge=1)

    mainroad: bool
    guestroom: bool
    basement: bool
    hotwaterheating: bool
    airconditioning: bool

    parking: int = Field(..., ge=0)
    prefarea: bool

    furnishingstatus: Literal[
        "furnished",
        "semi-furnished",
        "unfurnished"
    ]


class PredictionResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    currency: str
    formatted: str
    model: str