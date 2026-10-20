# [{"clusterId": 0, "size": 121996, "label": "Took or threatened to take negative or legal action / Threatened or suggested your credit would be damaged", "topProducts": [{"name": "Debt collection", "share": 1.0}], "topIssues": [{"name": "Took or threatened to take negative or legal action", "share": 1.0}], "topSubIssues": [{"name": "Threatened or suggested your credit would be damaged", "share": 0.846}, {"name": "Threatened to sue you for very old debt", "share": 0.053}, {"name": "Sued you without properly notifying you of lawsuit", "share": 0.039}], "topCompanies": [{"name": "ENCORE CAPITAL GROUP INC.", "share": 0.063}, {"name": "Resurgent Capital Services L.P.", "share": 0.055}, {"name": "Portfolio Recovery Associates, LLC", "share": 0.055}], "representativeIds": [25203506, 24980495, 24003880, 20235104, 23903491], "firstDate": "2023-01-01", "lastDate": "2026-10-10"},
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import date

class Cluster(BaseModel):
    clusterId: int = Field(..., description="The unique identifier for the cluster")
    size: int = Field(..., description="The number of complaints in the cluster")
    label: str = Field(..., description="A human-readable label describing the cluster")
    topProducts: list[dict] = Field(..., description="A list of top products associated with the cluster, each with a name and share")
    topIssues: list[dict] = Field(..., description="A list of top issues associated with the cluster, each with a name and share")
    topSubIssues: list[dict] = Field(..., description="A list of top sub-issues associated with the cluster, each with a name and share")
    topCompanies: list[dict] = Field(..., description="A list of top companies associated with the cluster, each with a name and share")
    representativeIds: list[int] = Field(..., description="A list of representative complaint IDs for the cluster")
    firstDate: date = Field(..., description="The date of the first complaint in the cluster in YYYY-MM-DD format")
    lastDate: date = Field(..., description="The date of the last complaint in the cluster in YYYY-MM-DD format")