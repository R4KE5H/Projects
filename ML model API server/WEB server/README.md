System Details:
OS:  Windows 11
Processor: 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz   2.50 GHz
RAM: 8.00 GB
Architecture: 64-bit operating system, x64-based processor
FPS: 6-7 fps


The application is designed in such a way that it runs an ML model in Multi-Camera to detect the car and track the specified object in the Django framework. Use the “setting.json” file to add the camera path and ml models to respective dictionary keys.

Each “Streaming()” object is initiated using threading and for each camera, required weights have been loaded via the “setting.json” file.

![alt text](https://github.com/R4KE5H/Projects/blob/main/Multi%20Camera%20Threading/app/demo/Multi%20Camera%20Threading%20Flowchart.png)




Language Used:
Python 3, HTML

Links:
Projects/Multi Camera Threading at main · R4KE5H/Projects (github.com)

