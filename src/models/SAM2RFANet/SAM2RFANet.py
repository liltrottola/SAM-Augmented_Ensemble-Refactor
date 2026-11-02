import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from decoders import RFA

# class Adapter(nn.Module):
#     def __init__(self, blk) -> None:
#         super(Adapter, self).__init__()
#         self.block = blk
#         dim = blk.attn.qkv.in_features
#         self.prompt_learn = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.GELU(),
#             nn.Linear(32, dim),
#             nn.GELU()
#         )
#
#     def forward(self, x):
#         prompt = self.prompt_learn(x)
#         promped = x + prompt
#         net = self.block(promped)
#         return net
    
  #adapter(1)
class Adapter(nn.Module):
    def __init__(self, blk, num_heads=4) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        head_dim = dim // num_heads
        self.num_heads = num_heads

        # 多头prompt
        self.prompt_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, 32),
                nn.GELU(),
                nn.Linear(32, head_dim),
                nn.GELU()
            ) for _ in range(num_heads)
        ])
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多头处理
        x_split = torch.chunk(x, self.num_heads, dim=-1)
        prompts = [head(xh) for head, xh in zip(self.prompt_heads, x_split)]
        prompt = torch.cat(prompts, dim=-1)
        # 门控调节
        gate_weight = self.gate(x)
        prompt = prompt * gate_weight
        promped = x + prompt
        net = self.block(promped)
        return net


class SAM2RFANet(nn.Module):
    def __init__(self, checkpoint_path=None,num_classes=1,channels=[1152,576,288,144], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6',input_resolution=[11,22,44,88]) -> None:
        super(SAM2RFANet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.decoder = RFA(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                             dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)

        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        dec_outs = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        return p1, p2, p3, p4


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2RFANet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2,_ = model(x)
        print(model)
