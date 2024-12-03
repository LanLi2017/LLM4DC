from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_wf_user import choose_ops

app = FastAPI()

# Pydantic model for request body
class OpsPoolRequest(BaseModel):
    user_choice: str
    ops_pool: list

@app.post("/choose")
async def choose(input_data: OpsPoolRequest):
    try:
        result = choose_ops(input_data.user_choice, input_data.ops_pool)
        return {"success": True, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
