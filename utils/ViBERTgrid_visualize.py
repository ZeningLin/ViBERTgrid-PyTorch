import math
import torch
import matplotlib.pyplot as plt


def ViBERTgrid_visualize(ViBERTgrid: torch.Tensor) -> None:
    num_pic = ViBERTgrid.shape[0]
    width = int(math.sqrt(num_pic))
    height = int(num_pic / width)
    grid_convert = torch.mean(ViBERTgrid, dim=1)

    plt.figure()
    for w in range(width):
        for h in range(height):
            pic_index = w * width + h + 1
            if pic_index <= num_pic:
                plt.subplot(width, height, pic_index)
                curr_pic = grid_convert[pic_index - 1].detach().numpy()
                plt.imshow(curr_pic)
    plt.show()
