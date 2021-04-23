from typing import Optional

from fastapi import FastAPI

from radie.src.extractor import Extractor

app = FastAPI()

extractor = Extractor(do_preprocessing=True,
                      do_split_sentence=True,
                      do_tokenize=True)


@app.get("/test")
async def root():
    return {"message": "Hello World !"}


@app.get("/radie/main")
async def main(report: str):
    return extractor(report)


@app.get("/radie/ner")
async def ner(report: str):
    return extractor.ner(report)


@app.get("/radie/clinical_object")
async def clinical_object(report: str, target_entity: Optional[str] = None):
    return extractor.focus_clinical_object(report, target_entity)