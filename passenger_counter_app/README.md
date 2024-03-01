# Intel OpenVino Passenger Counter Application

### Prerequisites : 

1) Ubuntu 18.04
2) Python 3.6
3) OpenVino toolkit 2020.3


#### Install and setup all the dependencies :
```
cd <path_to_the_people-counter-python_directory>
./setup.sh
```
make sure node and npm versions are 6.17.1 and 3.10.10
```
node -v
npm -v 
```
If the versions are different run below command to fix it:
```
sudo npm install -g n
sudo n 6.17.1
```
#### Now clone :
```
clone : https://github.com/Sid01mslp/passenger_counter_app
sudo apt-get update && sudo apt-get install git
```

#### install other dependencies &Go to the application directory:

### run this commands for installing mosca server: 

cd webservice/server
npm install
npm i jsonschema@1.2.6

#### run below code for webserver: 
```
cd ../ui
npm install
```

#### The config file:
```
resource/config.json
```
can set any video or webcam from here: 

Example of the config.json file:<br>
```
{

    "inputs": [
	    {
            "video": "videos/video1.mp4"
        }
    ]
}
```

### Now Run the applications step: 

i)open mosca 
```
1) sudo lsof -i:3000
2) cd webservice/server/node-server
3) node ./server.js
```

ii)open the gui
```
1) cd ../../ui
2) npm run dev
```
iii) openning ffpeg server
```
1) cd ../..
2) sudo ffserver -f ./ffmpeg/server.conf
```
iv) run using python script
```
1) source /opt/intel/openvino/bin/setupvars.sh
2) source openvino_env/bin/activate
3) cd ./applications
4) python3 people_counter.py -m /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```

#### Using camera instead of video file
```
For example:
```
python3.5 main.py -i CAM -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:8090/fac.ffm
```
run this port in web browser: http://localhost:8080/
**Note:**
If there camera can not be able to show the full resolution just change the video size according to the resolution of camera.
```
