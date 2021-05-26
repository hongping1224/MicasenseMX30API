# MicasenseMX30API
API for Micasense MX30 camera


## install dependency:

run install_dependency.bat to install dependency

```
install_dependency.bat
```


## 1. 遙測影像 計算matrix ./calallignment

allow method : POST

->input  {  "1":band 1 link,
            "2":band 2 link,
            "3":band 3 link,
            "4":band 4 link,
            "5":band 5 link,
            "maxiteration" : int}

Note: maxiteration is optional, default is 20

Example:
```
1:https://tmpfiles.org/dl/158007/IMG_0078_1.tif
2:https://tmpfiles.org/dl/158008/IMG_0078_2.tif
3:https://tmpfiles.org/dl/158009/IMG_0078_3.tif
4:https://tmpfiles.org/dl/158010/IMG_0078_4.tif
5:https://tmpfiles.org/dl/158011/IMG_0078_5.tif
maxiteration:20

./calallignment?1=https://tmpfiles.org/dl/158007/IMG_0078_1.tif&2=https://tmpfiles.org/dl/158008/IMG_0078_2.tif&3=https://tmpfiles.org/dl/158009/IMG_0078_3.tif&4=https://tmpfiles.org/dl/158010/IMG_0078_4.tif&5=https://tmpfiles.org/dl/158011/IMG_0078_5.tif&maxiteration=20
```
->return {  "allignmat": a 5x3x3 array}

Example:
```
{'allignmat':[[[1.0018138885498047, -0.005332774017006159, 30.765457153320312], [0.006256055552512407, 1.00254225730896, 34.800621032714844], [1.5525171193075948e-06, 7.275947382368031e-07, 1.0]], [[0.9956827759742737, -0.0012916148407384753, 42.02390670776367], [0.0014379125786945224, 0.9979942440986633, 8.522543907165527], [-1.1532963526406093e-06, 7.530898642471584e-07, 1.0]], [[0.995963454246521, -0.001184804947115481, 23.758642196655273], [0.0006243085372261703, 0.9972562193870544, 11.241520881652832], [-1.5542806295343325e-06, 2.1815537820657482e-07, 1.0]], [[1.0034475326538086, -0.0028804221656173468, 6.537084102630615], [0.00292379898019135, 1.0032029151916504, 42.04877471923828], [1.3338612916413695e-06, -4.221681138005806e-07, 1.0]], [[1.000811219215393, 0.001381688634864986, -0.3662553131580353], [-0.001451150281354785, 0.9998471736907959, 0.3415140211582184], [9.58449959398422e-07, -8.217737104132539e-07, 1.0]]]}
```

## 2. 近景攝影 計算matrix ./calallignment2 

allow method : POST

->input  {  "1":band 1 link,
            "2":band 2 link,
            "3":band 3 link,
            "4":band 4 link,
            "5":band 5 link,
            }


Example:
```
1:https://tmpfiles.org/dl/158007/IMG_0078_1.tif
2:https://tmpfiles.org/dl/158008/IMG_0078_2.tif
3:https://tmpfiles.org/dl/158009/IMG_0078_3.tif
4:https://tmpfiles.org/dl/158010/IMG_0078_4.tif
5:https://tmpfiles.org/dl/158011/IMG_0078_5.tif


./calallignment?1=https://tmpfiles.org/dl/158007/IMG_0078_1.tif&2=https://tmpfiles.org/dl/158008/IMG_0078_2.tif&3=https://tmpfiles.org/dl/158009/IMG_0078_3.tif&4=https://tmpfiles.org/dl/158010/IMG_0078_4.tif&5=https://tmpfiles.org/dl/158011/IMG_0078_5.tif
```
->return {  "allignmat": a 5x3x3 array}

Example:
```
{'allignmat':[[[1.0018138885498047, -0.005332774017006159, 30.765457153320312], [0.006256055552512407, 1.00254225730896, 34.800621032714844], [1.5525171193075948e-06, 7.275947382368031e-07, 1.0]], [[0.9956827759742737, -0.0012916148407384753, 42.02390670776367], [0.0014379125786945224, 0.9979942440986633, 8.522543907165527], [-1.1532963526406093e-06, 7.530898642471584e-07, 1.0]], [[0.995963454246521, -0.001184804947115481, 23.758642196655273], [0.0006243085372261703, 0.9972562193870544, 11.241520881652832], [-1.5542806295343325e-06, 2.1815537820657482e-07, 1.0]], [[1.0034475326538086, -0.0028804221656173468, 6.537084102630615], [0.00292379898019135, 1.0032029151916504, 42.04877471923828], [1.3338612916413695e-06, -4.221681138005806e-07, 1.0]], [[1.000811219215393, 0.001381688634864986, -0.3662553131580353], [-0.001451150281354785, 0.9998471736907959, 0.3415140211582184], [9.58449959398422e-07, -8.217737104132539e-07, 1.0]]]}
```


## 3. 用matrix 做fusion ./allignment

allow method : POST

input  {  "1":band 1 link,
            "2":band 2 link,
            "3":band 3 link,
            "4":band 4 link,
            "5":band 5 link,
            "allignmat": result from 1. }
Example:
```
1:https://tmpfiles.org/dl/155978/IMG_0078_1.tif
2:https://tmpfiles.org/dl/155979/IMG_0078_2.tif
3:https://tmpfiles.org/dl/155980/IMG_0078_3.tif
4:https://tmpfiles.org/dl/155981/IMG_0078_4.tif
5:https://tmpfiles.org/dl/155982/IMG_0078_5.tif

allignmat:[[[1.0018138885498047, -0.005332774017006159, 30.765457153320312], [0.006256055552512407, 1.00254225730896, 34.800621032714844], [1.5525171193075948e-06, 7.275947382368031e-07, 1.0]], [[0.9956827759742737, -0.0012916148407384753, 42.02390670776367], [0.0014379125786945224, 0.9979942440986633, 8.522543907165527], [-1.1532963526406093e-06, 7.530898642471584e-07, 1.0]], [[0.995963454246521, -0.001184804947115481, 23.758642196655273], [0.0006243085372261703, 0.9972562193870544, 11.241520881652832], [-1.5542806295343325e-06, 2.1815537820657482e-07, 1.0]], [[1.0034475326538086, -0.0028804221656173468, 6.537084102630615], [0.00292379898019135, 1.0032029151916504, 42.04877471923828], [1.3338612916413695e-06, -4.221681138005806e-07, 1.0]], [[1.000811219215393, 0.001381688634864986, -0.3662553131580353], [-0.001451150281354785, 0.9998471736907959, 0.3415140211582184], [9.58449959398422e-07, -8.217737104132539e-07, 1.0]]]

./allignment?1=https://tmpfiles.org/dl/158007/IMG_0078_1.tif&2=https://tmpfiles.org/dl/158008/IMG_0078_2.tif&3=https://tmpfiles.org/dl/158009/IMG_0078_3.tif&4=https://tmpfiles.org/dl/158010/IMG_0078_4.tif&5=https://tmpfiles.org/dl/158011/IMG_0078_5.tif&allignmat=[[[1.0018361806869507, -0.0053227986209094524, 30.735538482666016], [0.006259461399167776, 1.002570390701294, 34.75965881347656], [1.5659481960028643e-06, 7.520987423959014e-07, 1.0]], [[0.995696485042572, -0.0013030744157731533, 41.99313735961914], [0.00144621089566499, 0.9979988932609558, 8.509234428405762], [-1.136687615144183e-06, 7.421792247441772e-07, 1.0]], [[0.9959765076637268, -0.001188539550639689, 23.739185333251953], [0.0006357940146699548, 0.9972655773162842, 11.220815658569336], [-1.5398014738821075e-06, 2.11923406823189e-07, 1.0]], [[1.0034830570220947, -0.0028713990468531847, 6.522168159484863], [0.002935728058218956, 1.0032304525375366, 41.9962043762207], [1.358306008114596e-06, -4.046684409786394e-07, 1.0]], [[1.000811219215393, 0.0013820487074553967, -0.365969181060791], [-0.001450772164389491, 0.9998471736907959, 0.34115830063819885], [9.59199269345845e-07, -8.226305681091617e-07, 1.0]]]

```

return {  "1":band 1 link,
            "2":band 2 link,
            "3":band 3 link,
            "4":band 4 link,
            "5":band 5 link,

Example:
```
{
    "1": "/download/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_1.tif",
    "2": "/download/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_2.tif",
    "3": "/download/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_3.tif",
    "4": "/download/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_4.tif",
    "5": "/download/d1f58a02-3e55-4f8f-b5f7-f801e6a967e9_5.tif"
}
```

## 4. 用fusion 後的影像計算指標或疊合 ./cal

allow method : POST


input  {   "1":band 1 link,
            "2":band 2 link,
            "3":band 3 link,
            "4":band 4 link,
            "5":band 5 link,
            "ops": ["nbi","ndvi","cir","rgb","tgi"] }

Example:
```
ops:["nbi","ndvi","cir","rgb","tgi"]
1:http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_1.tif
2:http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_2.tif
3:http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_3.tif
4:http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_4.tif
5:http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_5.tif

./cal?ops=["nbi","ndvi","cir","rgb","tgi"]&1=http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_1.tif&2=http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_2.tif&3=http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_3.tif&4=http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_4.tif&5=http://13.75.68.36:5000/download/81033cea-2f87-4c29-954f-3a73badabdc2_5.tif
```

return {   "ndvi":ndvi result link,
            "rgb":rgb result link,
            "nbi":nbi result link,
            "cir":cir result link,
            "tgi":tgi result link,

Example:
```
{
    "ndvi": "/download/79373fd6-0838-40ce-8948-bba0340eedec_ndvi.tif",
    "rgb": "/download/79373fd6-0838-40ce-8948-bba0340eedec_rgb.png",
    "nbi": "/download/79373fd6-0838-40ce-8948-bba0340eedec_nbi.tif",
    "cir": "/download/79373fd6-0838-40ce-8948-bba0340eedec_cir.png",
    "tgi": "/download/79373fd6-0838-40ce-8948-bba0340eedec_tgi.tif"
}
```

## 5. 下載或刪除cache影像 ./download/<filename>

allow method : DELETE, GET, POST

input {}

Example:
```
http://13.75.68.36:5000/download/79373fd6-0838-40ce-8948-bba0340eedec_ndvi.tif
```

return {message:"message"}

Example

```
{'message':'done'}
```


