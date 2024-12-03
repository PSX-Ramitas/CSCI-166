
import armNN
from armNN import torch
from torchview import draw_graph

import armSim
from matplotlib import pyplot as plt

env = armSim.ArmEnv()

n_action = env.actionSpace

# setup which device to use
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

model = armNN.ArmDQN(n_action).to(device)

path = 'Models/2024-12-02/Test03_04:08:10.pth'

model.load_state_dict(torch.load(path))

model.eval()

env.connect()
env.load_environment()

state1, state2 = env.resetSim()
state1 = torch.tensor(state1, dtype=torch.float32, device=device).unsqueeze(0)
state1 = state1.permute(0, 3, 2, 1)
state2 = torch.tensor(state2, dtype=torch.float32, device=device).unsqueeze(0)

for i in range(300):
    action = None

    with torch.no_grad():
        action = model(state1, state2).max(1).indices.view(1, 1)
        print(action)
    
    array = env.convInttoBase3(action.item())

    print(f"Action Index: {action.item()}")
    print(f"Action Array: {array}")
    print("")
    
    observation, _, _ = env.takeAction(action.item())
    state1 = observation[0]
    state2 = observation[1]
    state1 = torch.tensor(state1, dtype=torch.float32, device=device).unsqueeze(0)
    state1 = state1.permute(0, 3, 2, 1)
    state2 = torch.tensor(state2, dtype=torch.float32, device=device).unsqueeze(0)

    # Display the image using Matplotlib
    # plt.imshow(observation[0])
    # plt.axis('off')  # Hide axes for better visualization
    # plt.show()


# Generate computation graph
#graph = draw_graph(model, input_data=(state1, state2), expand_nested=True)
#graph.visual_graph.view()  # Opens the visual diagram

env.disconnect()
