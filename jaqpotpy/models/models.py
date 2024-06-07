
from pydantic import BaseModel
from typing import List, Optional

class MetaInfo(BaseModel):
    identifiers: Optional[List[str]]
    comments: Optional[List[str]]
    descriptions: Optional[List[str]]
    titles: Optional[List[str]]
    subjects: Optional[List[str]]
    publishers: Optional[List[str]]
    creators: Optional[List[str]]
    contributors: Optional[List[str]]
    audiences: Optional[List[str]]
    rights: Optional[List[str]]
    sameAs: Optional[List[str]]
    seeAlso: Optional[List[str]]
    hasSources: Optional[List[str]]
    doi: Optional[List[str]]
    date: Optional[str]
    picture: Optional[str]
    markdown: Optional[str]
    tags: Optional[List[str]]
    read: Optional[List[str]]
    write: Optional[List[str]]
    execute: Optional[List[str]]

class Doa(BaseModel):
    meta: Optional[MetaInfo]
    modelId: Optional[str]
    doaMatrix: Optional[List[List[float]]]
    aValue: Optional[float]

