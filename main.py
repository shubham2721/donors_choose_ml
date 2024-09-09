from fastapi import FastAPI
import joblib
import pandas as pd
import final_inference as fi
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the pre-trained ML model
# model = joblib.load("lr.joblib")
#Define a POST endpoint for making predictions 

def predict(body: dict):
    resource_data = body.resource_data.dict()  # Convert to dictionary
    resource_df = pd.DataFrame([resource_data])
    
    # Extract product data
    product_data = body.product
    product_df = pd.DataFrame([p.dict() for p in product_data])
    
    # Call your prediction function
    predicted_value = fi.predict_score_inf(resource_df, product_df)

    if predicted_value[0][0] == 0:
        out = 'Project Cannot be approved'
    else:
        out = 'Project Approved'
    prob = round(predicted_value[1][0] * 100.0,2)
    return out, prob

# Define a model for the product
class Product(BaseModel):
    id: str
    description: str
    quantity: int
    price: float

class ResourceData(BaseModel):
    Unnamed_0: int
    id: str
    teacher_id: str
    teacher_prefix: str
    school_state: str
    project_submitted_datetime: str
    project_grade_category: str
    project_subject_categories: str
    project_subject_subcategories: str
    project_title: str
    project_essay_1: str
    project_essay_2: str
    project_essay_3: str
    project_essay_4: str
    project_resource_summary: str
    teacher_number_of_previously_posted_projects: int

# Define a model for the prediction input
class PredictionInput(BaseModel):
    resource_data: ResourceData
    product: List[Product]

# all the below code will be executed 
@app.get("/get_prediction")
async def read_root():
    return FileResponse(str("Data Form.html"))

@app.post("/predict")
# all the below code will be executed 
async def predict_endpoint(data: PredictionInput):
    print(data)
    prediction = predict(data)
    return prediction