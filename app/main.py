import os

from fastapi import FastAPI

from radie.src.utils.wrapper import Structured_Reporting

app = FastAPI()

base_path = '/data/sugimoto/experiments/outputs'
ner_path = os.path.join(base_path, 'b4e8a754bc784356b2ed16dce91878a1')
sc_path = os.path.join(base_path, '192d011ea1e84760a02f00460fa77f9a')
re_path = os.path.join(base_path, 'b424a480ea284606bc4f2d73c1385d9b')

sr = Structured_Reporting(ner_path=ner_path, sc_path=sc_path, re_path=re_path, do_certainty_completion=False,
                          do_preprocessing=True, do_split_sentence=True, do_tokenize=True)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.get("/ner")
# async def ner(params: str):
#     tokens = mc.parse(params).strip().split(' ')
#     result = tagger.predict(tokens)
#     return {"tokens": tokens, "tagged_result": result}

@app.get("/radie")
async def main(report: str):
    res = sr.structuring(report)
    res.chunking()
    return res.json(ensure_ascii=False)