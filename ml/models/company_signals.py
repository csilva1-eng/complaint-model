from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import date

class CompanySignal(BaseModel):
    company: str = Field(..., description="The name of the company")
    tier: Literal['alert', 'watch'] = Field(..., description="The tier of the signal, either 'alert' or 'watch'")
    weekStart: date = Field(..., description="The start date of the week in YYYY-MM-DD format")
    count: int = Field(..., description="The number of complaints for the company in that week")
    baselineMean: float = Field(..., description="The trailing average of complaints for the company")
    zScore: float = Field(..., description="The z-score indicating how many standard deviations the count is from the baseline mean")
    reason: str = Field(..., description="A human-readable explanation of the alert or watch status")