from __future__ import annotations
from pydantic import BaseModel, Field, condecimal


# Input validation schema for predictions
class PredictPayload(BaseModel):
    company_code: str = Field(..., min_length=1)
    document_number: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    payment_type: str = Field(..., min_length=1)
    amount: condecimal(gt=0) = Field(..., description="Positive amount")
    currency_code: str = Field(..., min_length=1, max_length=3)
    transaction_type: str = Field(..., min_length=1)


