import math
import torch
import torchvision.transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from typing import List, Any, Tuple
from collections import defaultdict


STANDARD_COLORS = [
    "AliceBlue",
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "BlanchedAlmond",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]


def ViBERTgrid_visualize(ViBERTgrid: Any) -> None:
    if isinstance(ViBERTgrid, (List, Tuple)):
        num_pic = len(ViBERTgrid)
    elif isinstance(ViBERTgrid, torch.Tensor):
        num_pic = ViBERTgrid.shape[0]
    width = int(math.sqrt(num_pic))
    height = int(num_pic / width)

    if isinstance(ViBERTgrid, (List, Tuple)):
        grid_convert = [
            50 * ViBERTgrid_.float().detach().numpy() for ViBERTgrid_ in ViBERTgrid
        ]
    elif isinstance(ViBERTgrid, torch.Tensor):
        grid_convert = torch.mean(ViBERTgrid.float(), dim=1).detach().numpy()
        grid_convert *= 255

    plt.figure()
    for w in range(width):
        for h in range(height):
            pic_index = w * width + h + 1
            if pic_index <= num_pic:
                plt.subplot(width, height, pic_index)
                curr_pic = grid_convert[pic_index - 1]
                plt.imshow(curr_pic)
    plt.show()


def inference_visualize(
    image: torch.Tensor,
    class_label: torch.Tensor,
    pred_ss: torch.Tensor,
    pred_mask: torch.Tensor,
) -> None:
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    class_label = class_label.detach().cpu().numpy()
    pred_ss = pred_ss.squeeze(0).permute(1, 2, 0).argmax(dim=2).detach().cpu().numpy()
    pred_mask = (
        pred_mask.squeeze(0).permute(1, 2, 0).argmax(dim=2).detach().cpu().numpy()
    )

    class_label *= 255
    pred_ss *= 255
    pred_mask *= 255

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("orig image")

    plt.subplot(2, 2, 2)
    plt.imshow(pred_ss)
    plt.title("pred segmentation")

    plt.subplot(2, 2, 3)
    plt.imshow(pred_mask)
    plt.title("pred pos neg")

    plt.subplot(2, 2, 4)
    plt.imshow(class_label)
    plt.title("ground truth")

    plt.show()


def draw_box(
    image: Any,
    boxes_dict_list: List[defaultdict],
    class_list: List[str],
    line_thickness: int = 4,
):
    assert isinstance(
        image, (PIL.Image, torch.Tensor)
    ), f"image must be PIL.Image or torch.Tensor, {type(image)} given"

    if isinstance(image, torch.Tensor):
        image = torchvision.transforms.ToPILImage()(image.cpu())

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for idx, curr_class_box in enumerate(boxes_dict_list):
        curr_color = STANDARD_COLORS[idx % len(STANDARD_COLORS)]
        curr_class_str = class_list[idx]
        text_width, text_height = font.getsize(curr_class_str)
        margin = np.ceil(0.05 * text_height)
        for text, coor in curr_class_box.items():
            left = coor[0]
            top = coor[1]
            right = coor[2]
            bottom = coor[3]

            text_bottom = top

            draw.line(
                [
                    (left, top),
                    (left, bottom),
                    (right, bottom),
                    (right, top),
                    (left, top),
                ],
                width=line_thickness,
                fill=curr_color,
            )

            draw.rectangle(
                [
                    (left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom),
                ],
                fill=curr_color,
            )
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                curr_class_str,
                fill="black",
                font=font,
            )

    image.save("./inference_result.jpg")


if __name__ == "__main__":
    from data.SROIE_dataset import load_train_dataset as SROIE_load
    from data.EPHOIE_dataset import load_train_dataset as EPHOIE_load
    from transformers import BertTokenizer
    from tqdm import tqdm

    dir_processed = r"dir_to_processed"
    model_version = "bert-base-chinese"
    print("loading bert pretrained")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_loader, val_loader = EPHOIE_load(
        dir_processed, batch_size=4, num_workers=0, tokenizer=tokenizer
    )

    total_loss = 0
    num_batch = len(train_loader)
    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        ViBERTgrid_visualize(class_label)
        ViBERTgrid_visualize(pos_neg)
