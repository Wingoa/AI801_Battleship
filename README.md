# AI801_Battleship

Battleship agent for AI801

---

Command in Docker terminal to install files within our volume we created for the class

#docker run -p 8889:8888 --mount type=volume,source=ai801-workspace,target=/opt/aima-python -it --rm cbw5803/aima-devbox:latest bash -c "pip install gymnasium && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"

---

1. Run Train.py

This will create and agent for a 10x10 board

2. Run battleship_game.py

This will run the dashboard. However to run the model that was just trained, the pygame file will need to be pointed at the newly 10x10 model.

The initialization of the BattleShioEnv in the pygame will also need to be changed to match the 10x10 model, otherwise it will run our 6x6 version of the model which had better success than the 10x10.
