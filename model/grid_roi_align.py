import torch
import torch.nn as nn
import torchvision

from typing import Tuple, Any


class GridROIAlign(nn.Module):
    """apply ROIAlign* to the P_fuse feature map

    (*He et al. Mask R-CNN. ICCV, 2017.)

    Parameters
    ----------
    output_size : int or Tuple[int, int], optional
        shape of aligned ROI, in form (H, W), can either be int or Tensor.
        if int, output_W = output_H = output_size, 
        if tensor, output_H = output_size[0], output_W = output_size[1]
        by default 7
    step : int, optional
        downsampling rate of the P_fuse feature map, by default 4
    output_reshape : bool, optional
        if True, output shape = (N, seqLen, C, output_H, output_W)
        else (N * seqLen, C, output_H, output_W),
        by default False.  
        if mask is not None, no reshape will be applied.

    """
    def __init__(
        self, 
        output_size: Any = 7, 
        step: int = 4, 
        output_reshape: bool = False
    ) -> None:
        super().__init__()
        
        if isinstance(output_size, int) or isinstance(output_size, Tuple):    
            self.output_size = output_size
        else:
            raise TypeError(f'parameter \'output_size\' requires int or tuple, {type(output_size)} were given')

        self.spatial_scale = 1 / float(step)
        self.output_reshape = output_reshape
        
        self.ROI_Align_net = torchvision.ops.RoIAlign(
            output_size=self.output_size, 
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1
        )
        
    def forward(
        self,
        feature_map: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """forward propagation of GridROIAlign

        Parameters
        ----------
        feature_map : torch.Tensor
            P_fuse feature map from the ResNet18-FPN network, in shape (N, C, H, W)
        coords : torch.Tensor
            coordinates tensor directly from SROIEDataset, in shape (N, seqLen, 4)
        mask : torch.Tensor, optional
            mask tensor from SROIEDataset, in shape (N, seqLen), by default None
            
        Returns
        -------
        roi_output : torch.Tensor
            aligned ROI, 
            if reshaped, return a Tensor (N, seqLen, C, output_H, output_W)
            else (N * seqLen, C, output_H, output_W)
        
        """
        coords_list = list()
        if mask is None:
            for bs_index in range(coords.shape[0]):
                coords_list.append(coords[bs_index])
        else:
            self.output_reshape = False
            for bs_index in range(coords.shape[0]):
                # get the length of valid corpus, discard paddings
                valid_index = mask[bs_index].sum(dim=0).int()
                coords_list.append(coords[bs_index, : valid_index])

        roi_output: torch.Tensor = self.ROI_Align_net(
            feature_map,
            coords_list
        )

        if self.output_reshape:
            if isinstance(self.output_size, Tuple):
                output_h = self.output_size[0]
                output_w = self.output_size[1]
            else:
                output_h = self.output_size
                output_w = self.output_size
            roi_output = roi_output.reshape(
                coords.shape[0], -1,
                coords.shape[1], output_h, output_w
            )

        return roi_output


if __name__ == '__main__':
    """
    torchvision.ops.roi_align的使用:
    输入:
        input: Tensor, (N, C, H, W), 
            要进行ROI align的feature map
        boxes: Tensor | List[Tensor], 
            两种输入方式
            方式一:
                Tensor形式输入，尺寸(K, 5)，K是待处理的ROI总数，5中后四个是坐标，第一个是index，表示该坐标
                是它所在的batch中的第几个坐标
            方式二:
                List[Tensor]的形式输入，list中的每个元素是一个Tensor，对应每一个batch中要提取的ROI坐标
                每一个Tensor尺寸为(K, 4)，K为这个batch中的ROI总数，4对应四个坐标，左上xy和右下xy
        output_size: Any, 
            最终提取出来的ROI的高和宽，可以是int（正方形）或者(int, int)（矩形）的形式
        spatial_scale: float = 1, 
            将输入坐标映射到框坐标的比例因子
            假设原始图像为224*224，输入的boxes也是这个尺度上的坐标
            但是经过特征提取可能获取的feature map大小为112*112，此时这个spatial_scale值应当设置为0.5
        sampling_ratio: int = -1, 
            插值网格中用于计算每个合并输出bin的输出值的采样点数目。
            如果> 0，则使用sampling_ratio x sampling_ratio个网格点。
            如果<= 0，则使用自适应数量的网格点(计算为cell (roi_width / pooled_w)，同样计算高度)
        aligned: bool = False
    输出:
        output: Tensor, (TN, C, output_size, output_size)
            TN是boxes中给出的bbox坐标总数
            如果输入的坐标是重复的，这个函数不会自动检测出来，输入多少组坐标就输出多少个ROI Align结果

    """

    input = torch.arange(60000, dtype=torch.float32,
                         device='cuda').reshape(2, 3, 100, 100)
    coords = torch.tensor([[[11.4, 12.3, 54.1, 54.1], [11.4, 12.3, 54.1, 54.1], [0, 0, 0, 0]],
                          [[24.1, 34.1, 56.7, 56.7], [34.1, 4.1, 56.7, 7.1], [24.1, 14.1, 33, 96.7]]],
                          dtype=torch.float32, device='cuda')
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device='cuda')
    
    grid_roi_align = GridROIAlign(output_size=7, step=4, output_reshape=False)
    
    roi_output = grid_roi_align(input, coords, mask)

    print(input.shape, '\n', mask.shape, '\n',
          coords.shape, '\n', roi_output.shape)
