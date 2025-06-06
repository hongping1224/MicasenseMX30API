python -m pip install --upgrade pip

python -m pip install scikit-image
python -m pip install opencv-python
python -m pip install numpy
python -m pip install matplotlib
python -m pip install pysolar
python -m pip install mapboxgl
python -m pip install pytz
python -m pip install packaging
python -m pip install tifffile
python -m pip install imagecodecs
python -m pip install requests
python -m pip install flask
python -m pip install flask_cors
cd pyexiftool
python setup.py install --user
cd ..

setx exiftoolpath "%cd%\exiftool\exiftool.exe"
