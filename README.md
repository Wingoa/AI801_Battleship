# AI801_Battleship
Battleship agent for AI801


-------------------------

Command in Docker terminal to install files within our volume we created for the class

#docker run -p 8889:8888 --mount type=volume,source=ai801-workspace,target=/opt/aima-python -it --rm cbw5803/aima-devbox:latest bash -c "pip install gymnasium && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"


-------------------------
