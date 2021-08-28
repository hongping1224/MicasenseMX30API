from operator import truediv
import shutil
import requests 
import uuid
import os

tmpfolder = 'tmp'

if not os.path.exists(tmpfolder):
    os.makedirs(tmpfolder)

def GenerateRandomName():
    return os.path.join(tmpfolder,str(uuid.uuid4())+".tif")

def downloadImage(image_url):
    filename = GenerateRandomName()
        
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)
    
    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')
        return filename , False
    return filename ,True