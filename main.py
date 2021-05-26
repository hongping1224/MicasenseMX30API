from Allignment import AllignImage, GetAllignmentMatrix, SaveAllignmentMatrix, ReadAllignmentMatrix, allignmentMatrixTostring, loadfromstring
import cv2
import numpy as np
import json
import micasense.capture as cap
import os
import io
from ops import R,B,G,NIR,REDEDGE , RGB,CIR,NDVI,NBI,TGI
from downloadImage import downloadImage,GenerateRandomName,tmpfolder
from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS ,cross_origin

from numpy.core.records import fromstring
import requests
app = Flask(__name__)
CORS(app)
allignmentKey = 'allignmat'
np.set_printoptions(suppress=True)


@app.route('/calallignment', methods=['POST'])
#@cross_origin()
def calallignment():
    paths = []
    try:
        keys = ['1', '2', '3', '4', '5']
        for k in keys:
            if k not in request.values:
                return Response(f"{{'message':'Key \'{k}\' not Found'}}", status=400, mimetype='application/json')
            paths.append(downloadImage(request.values[k]))
        iteration = 20
        if "maxiteration" in request.values:
            try:
                iteration = int(request.values["maxiteration"])
            except:
                return Response(f"{{'message':'Key maxiteration must be a int'}}", status=400, mimetype='application/json')
        capture = cap.Capture.from_filelist(paths)
        allignmat = GetAllignmentMatrix(capture,iteration=iteration)
        s = allignmentMatrixTostring(allignmat)
    except:
        return Response('{"message":"something went wrong"}', status=400, mimetype='application/json')
    finally:
        clearCache(paths)
    return Response(f"{{'{allignmentKey}':{s}}}", status=200, mimetype='application/json')


@app.route('/calallignment2', methods=['POST'])
#@cross_origin()
def calallignment2():
    paths = []
    try:
        keys = ['1', '2', '3', '4', '5']
        for k in keys:
            if k not in request.values:
                return Response(f"{{'message':'Key \'{k}\' not Found'}}", status=400, mimetype='application/json')
            paths.append(downloadImage(request.values[k]))

        capture = cap.Capture.from_filelist(paths)
        allignmat = GetAllignmentMatrix2(capture)
        s = allignmentMatrixTostring(allignmat)
    except:
        return Response('{"message":"something went wrong"}', status=400, mimetype='application/json')
    finally:
        clearCache(paths)
    return Response(f"{{'{allignmentKey}':{s}}}", status=200, mimetype='application/json')



@app.route('/allignment', methods=['POST'])
#@cross_origin()
def allignImage():
    paths = []
    a = request.values[allignmentKey]
    a = json.loads(a)
    allignmat = loadfromstring(a)
    try:
        keys = ['1', '2', '3', '4', '5']
        for k in keys:
            if k not in request.values:
                return Response(f"{{'message':'Key \'{k}\' not Found'}}", status=400, mimetype='application/json')
            paths.append(downloadImage(request.values[k]))

        capture = cap.Capture.from_filelist(paths)

        im_allign = AllignImage(allignmat, capture)
        print(im_allign.shape)

        filename = GenerateRandomName()
        print(filename)
        k = {}
        for i in range(len(keys)):
            tmps = filename.replace(".tif",f"_{i+1}.tif")
            cv2.imwrite(tmps, im_allign[:,:,i])
            k[i+1] = "/download/"+os.path.basename(tmps)
        f = os.path.basename(filename)
        return Response(json.dumps(k), status=200, mimetype='application/json')
    except:
        return Response("{'message':'something went wrong!'}", status=400, mimetype='application/json')
    finally:
        clearCache(paths)


@app.route('/download/<filename>',methods = ['DELETE', 'GET', 'POST'])
#@cross_origin()
def download_file(filename):
    file_path = os.path.join(tmpfolder,filename)
    if os.path.isfile(file_path) ==False:
        return Response("{'message':'file not exist!'}", status=410, mimetype='application/json')
    if request.method == 'GET' or request.method == 'POST':
        return_data = io.BytesIO()
        with open(file_path, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)
        return send_file(return_data, mimetype='application/tif',
                    attachment_filename=filename)
    elif request.method =='DELETE':
        os.remove(file_path)
        return Response("{'message':'done'}", status=200, mimetype='application/json')
    return Response("{'message':'Method Not Allowed!'}", status=405, mimetype='application/json')

def clearCache(paths):
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


@app.route('/cal')
#@cross_origin()
def cal():
    if "ops" not in request.values:
        return Response(f"{{'message':'ops not Found'}}", status=400, mimetype='application/json')

    ops = json.loads(request.values['ops'])
    paths =[]
    try:
        keys = ['1', '2', '3', '4', '5']
        for k in keys:
            if k not in request.values:
                return Response(f"{{'message':'Key \'{k}\' not Found'}}", status=400, mimetype='application/json')
            paths.append(downloadImage(request.values[k]))
            
        a1 = cv2.imread(paths[0],cv2.IMREAD_LOAD_GDAL)
        a2 = cv2.imread(paths[1],cv2.IMREAD_LOAD_GDAL)
        a3 = cv2.imread(paths[2],cv2.IMREAD_LOAD_GDAL)
        a4 = cv2.imread(paths[3],cv2.IMREAD_LOAD_GDAL)
        a5 = cv2.imread(paths[4],cv2.IMREAD_LOAD_GDAL)
        im_allign = np.zeros((a1.shape[0],a1.shape[1],5), dtype=np.float32 )
        im_allign[:,:,B] = a1
        im_allign[:,:,G] = a2
        im_allign[:,:,R] = a3     
        im_allign[:,:,NIR] = a4     
        im_allign[:,:,REDEDGE] = a5     
        filename = GenerateRandomName()
        results = {}
        if "ndvi" in ops:
            ndvi = NDVI(im_allign)
            cv2.imwrite(filename.replace(".tif","_ndvi.tif"),ndvi)
            results["ndvi"] ="/download/"+ os.path.basename(filename.replace(".tif","_ndvi.tif"))
        if "rgb" in ops:
            rgb = RGB(im_allign)*255
            cv2.imwrite(filename.replace(".tif","_rgb.png"),rgb)
            results["rgb"] = "/download/"+ os.path.basename(filename.replace(".tif","_rgb.png"))
        if "nbi" in ops:
            nbi = NBI(im_allign)
            cv2.imwrite(filename.replace(".tif","_nbi.tif"),nbi)
            results["nbi"] = "/download/"+ os.path.basename(filename.replace(".tif","_nbi.tif"))
        if "cir" in ops:
            cir = CIR(im_allign)*255
            cv2.imwrite(filename.replace(".tif","_cir.png"),cir)
            results["cir"] = "/download/"+ os.path.basename(filename.replace(".tif","_cir.png"))
        if "tgi" in ops:
            tgi = TGI(im_allign)
            cv2.imwrite(filename.replace(".tif","_tgi.tif"),tgi)
            results["tgi"] = "/download/"+ os.path.basename(filename.replace(".tif","_tgi.tif"))
        return Response(json.dumps(results), status=200, mimetype='application/json')
    except Exception as e:
        print(e)
        return Response("{'message':'something went wrong!'}", status=400, mimetype='application/json')
    finally:
        clearCache(paths)

    

def main():
    app.debug = False
    app.run('0.0.0.0',port = 5500)
    return


if __name__ == "__main__":
    main()
