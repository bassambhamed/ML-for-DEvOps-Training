from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import json
from utils import preprocess,vectorize_log
import csv
from fastapi.responses import FileResponse,JSONResponse
import joblib
from pydantic import BaseModel
from io import BytesIO
#from fastapi.staticfiles import StaticFiles


#Define your FastAPI app
app = FastAPI()

# Mount static files at the specified path
#app.mount("/static", StaticFiles(directory="static"), name="static")

def predict(log):
    loaded_model = joblib.load('risky_safe_model.pkl')
    # Preprocess the log
    log = preprocess(log)
    # Vectorize the log using the model and reshape the vectorized log to be a 2D array 
    log = vectorize_log(log).reshape(1, -1)
    # Make prediction 
    prediction = loaded_model.predict(log)

    # Determine the response message based on the prediction
    if prediction[0] == 0:
        # Return the content directly
        return {"prediction": int(prediction[0]), "message": "Safe log"}
    else:
        return {"prediction": int(prediction[0]), "message": "Risky log"}


#Create an endpoint to use the saved model
@app.post("/predict")
def predict_func(log: str):
    result=predict(log)
    print("Result:", result)
    return JSONResponse(content=result)
    
#Create an endpoint to use the saved model
@app.post("/predict_csv_file")
async def respond(file: UploadFile = File(...)):
        
 # Read the CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(BytesIO(await file.read()))
        print('df renderrr!!!!',df)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    

    # Create an empty DataFrame to store the log_line and label
    newdf = df[['log_line', 'label']]
    # Define the number of rows to select for each label
    rows_per_label = 10

    # Use groupby and head to select two rows for each label
    selected_rows = pd.concat([
        newdf[newdf['label'] == 0].head(rows_per_label),
        newdf[newdf['label'] == 1].head(rows_per_label),
        newdf[newdf['label'] == 0].tail(rows_per_label),
        newdf[newdf['label'] == 1].tail(rows_per_label)
    ], ignore_index=True)

    # Display the resulting DataFrame
    print(selected_rows)
   
    # Make prediction 
    selected_rows['prediction'] = selected_rows['log_line'].apply(predict).apply(lambda d: d['prediction'] if isinstance(d, dict) and 'prediction' in d else None)

    print('newdf renderrr22222!!!!')
    print(selected_rows)
    print(selected_rows.columns)
   # Save the updated DataFrame to the Excel file
    output_filename = 'pretected_labels.xlsx'
    selected_rows.to_excel(output_filename, index=False)

    # Return the Excel file as a downloadable response
    #return FileResponse(output_filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=output_filename)
    return FileResponse(output_filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=output_filename)