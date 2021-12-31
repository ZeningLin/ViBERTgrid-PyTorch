import math
import torch
import matplotlib.pyplot as plt


def ViBERTgrid_visualize(ViBERTgrid: torch.Tensor) -> None:
    num_pic = ViBERTgrid.shape[0]
    width = int(math.sqrt(num_pic))
    height = int(num_pic / width)
    grid_convert = torch.mean(ViBERTgrid.float(), dim=1).detach().numpy()
    grid_convert = grid_convert * 255

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
    pred_mask = pred_mask.squeeze(0).permute(1, 2, 0).argmax(dim=2).detach().cpu().numpy()
    
    class_label *= 255
    pred_ss *= 255
    pred_mask *= 255
    
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("orig image")
    
    plt.subplot(2, 2, 2)
    plt.imshow(pred_ss)
    plt.title('pred segmentation')

    plt.subplot(2, 2, 3)
    plt.imshow(pred_mask)
    plt.title('pred pos neg')

    plt.subplot(2, 2, 4)
    plt.imshow(class_label)
    plt.title('ground truth')
    
    plt.show()


if __name__ == "__main__":
    from data.SROIE_dataset import load_train_dataset
    from transformers import BertTokenizer
    from tqdm import tqdm

    dir_processed = r"D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format\train"
    model_version = "bert-base-uncased"
    print("loading bert pretrained")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_loader, val_loader = load_train_dataset(
        dir_processed, batch_size=4, val_ratio=0.3, num_workers=0, tokenizer=tokenizer
    )

    total_loss = 0
    num_batch = len(train_loader)
    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        ViBERTgrid_visualize(class_label)
        ViBERTgrid_visualize(pos_neg)
