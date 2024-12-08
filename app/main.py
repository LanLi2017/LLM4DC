from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# app = FastAPI()
app = FastAPI(swagger_ui_parameters={"url": "./openapi.json"})


@app.post("/run")
def run_script(script_name: str):
    return {"script_output": ["foo", "bar"]}


@app.get("/operations")
async def get_operations(ops_pool):
    """
    Route to get the list of available operations.
    """
    return {"available_operations": ops_pool}


class UserChoiceRequest(BaseModel):
    user_choice: str
    ops_pool: list

@app.post("/choose")
async def choose(input_data: UserChoiceRequest):
    user_choice = input_data.user_choice
    ops_pool = input_data.ops_pool
    if user_choice not in ops_pool:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid choice. Available options: {ops_pool}"
        )
    return {"success": True, "result": {"selected_choice": user_choice}}