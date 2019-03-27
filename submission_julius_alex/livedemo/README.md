# Livedemo
This folder contains the scripts for the live demo we showed at the poster presentation.
## Usage
1. Forward the port you want to use via ssh.
2. Run the [Server](Server.py) script. Specify IP, port, the model path and the gpu you want to use.
3. Make sure that an Android smartphone is connected to the client computer and is accessible via adb.
4. Open the camera app on the Android smartphone. 
5. Run the [Client](Client.py) script and specify IP, port and the position of the camera shutter.
6. Press space to take a picture and "q" to quit.

## Example
client> ssh -L localhost:14441:localhost:14441 gpu \
server> python Server.py --ip localhost --port 14441 --model_path 'ssd/frozen_inference_graph.pb' --gpu 0 \
client> python Client.py --ip localhost --port 14441 --shutter_pos '540,1800'
  
