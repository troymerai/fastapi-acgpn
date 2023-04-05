from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from util import file
from clothes import clothes_unet
from human import openpose
from human import human_parsing
from tryon import acgpn
import cv2
import io

app = FastAPI()

filemanager = file.FileManager()
clothes_Unet = clothes_unet.Clothes_Unet(filemanager=filemanager)
#open_Pose = openpose.OpenPose(filemanager=filemanager)
human_Parsing = human_parsing.Human_Parsing(filemanager=filemanager)
acgpnModel = acgpn.ACGPN(filemanager=filemanager)

@app.get("/")
def root():
    return {"message": "Hello World"}

'''

@app.post("/clothes/")
async def inference_clothes(image: UploadFile = File(...)):
    contents = await image.read()
    # get clothes filename
    filename = filemanager.get_clothes_filename()  
    # RGB image load
    image = filemanager.bytes_image_open(contents)
    # save clothes image
    filemanager.save_clothes(image, filename) 
    msg = clothes_unet.predict(image, filename)
    return JSONResponse({'msg': msg, 'filename':filename})

@app.delete("/clothes/{filename}")
async def delete_clothes(filename: str):
    filemanager.remove_clothes(filename)
    return JSONResponse({'msg':"Delete"})

@app.post("/human/")
async def inference_human(image: UploadFile = File(...)):
    contents = await image.read()
    # get human filename
    filename = filemanager.get_human_filename() 
    # RGB image load
    image = filemanager.bytes_image_open(contents)
    # human image save
    filemanager.save_human(image, filename)
    # human parsing
    msg = human_Parsing.predict(image, filename)
    # human pose estimation
    #    msg = open_Pose.predict(image, filename)
    return JSONResponse({'msg':msg, 'filename':filename})

@app.delete("/human/{filename}")
async def delete_human(filename: str):
    filemanager.remove_human(filename)
    return JSONResponse({'msg':"Delete"})

@app.get("/result/")
async def tryon(c: str, h: str):
    output = acgpnModel.predict(c, h)
    img_str = cv2.imencode('.png', output)[1].tostring()
    return FileResponse(io.BytesIO(img_str), media_type='image/png')

'''