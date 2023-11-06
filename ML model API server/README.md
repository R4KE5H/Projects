System Details:
OS:  Windows 11
Processor: 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz   2.50 GHz
RAM: 8.00 GB
Architecture: 64-bit operating system, x64-based processor
FPS: 2-3 fps


The application is designed in such a way that it runs an ML model in Multi-Camera to detect the car and track the specified object in the Django framework. Use the “setting.json” file to add the camera path and ml models to respective dictionary keys.

Each “Streaming()” object is initiated using threading and for each camera, required weights have been loaded via the “setting.json” file.

The Flask server is deployed for “Yolov8” inference. “http://127.0.0.1:5000/yolov8n” API is integrated to run the inference in yolov8 model. Camera name and Frame are passed as post request and decoded to get the detection boxes and tracking ids. 

The Below diagram represents the flow of the application:



Language Used:
Python 3, HTML

Links:
Projects/ML model API server at main · R4KE5H/Projects (github.com)

