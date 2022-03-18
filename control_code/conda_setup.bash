#!/bin/bash
conda create --name cython-dev python=3.7 pip 
printf "\n\n%s\n\n%s\n\n        %s        \n        %s        \n\n%s\n" "Hi Drew, Hi Zack ;p" "Please run" "conda activate cython-dev" "pip install -r requirements.txt" "from the same folder as requirements.txt. If conda activate fails, follow the instructions it gives you for doing conda init." 
