import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block
from einops import rearrange

class CrossAttention_vv(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads):
        super(CrossAttention_vv, self).__init__()
        self.num_heads = num_heads
        self.k_dim = emb_dim
        self.v_dim = emb_dim

        self.proj_q1 = nn.Linear(in_channels, self.k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_channels, self.k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_channels, self.v_dim * num_heads, bias=False)
        self.proj_v2_ = nn.Linear(in_channels, self.v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(self.v_dim * num_heads, in_channels)
        self.proj_o_ = nn.Linear(self.v_dim * num_heads, in_channels)

    def forward(self, q, k, v1, v2, mask=None):
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v1 = rearrange(v1, 'b c h w -> b (h w) c')
        v2 = rearrange(v2, 'b c h w -> b (h w) c')

        batch_size, seq_len1, in_dim1 = k.size()
        seq_len2 = q.size(1)

        q1 = self.proj_q1(q).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(k).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v1 = self.proj_v2(v1).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        v2 = self.proj_v2_(v2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output1 = torch.matmul(attn, v1).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output1 = self.proj_o(output1)

        output2 = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output2 = self.proj_o(output2)
        return output1, output2

class MaCF(nn.Module):
    def __init__(self, in_channels, h=8, w=8):
        super(MaCF, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv_edge = nn.Conv2d(in_channels, in_channels // 2, kernel_size=7, stride=3)
        self.conv_img = nn.Conv2d(in_channels, in_channels // 2, kernel_size=7, stride=3)
        self.conv_img_v = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=2)
        self.conv_img_v2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=2)
        self.Adapt_v = nn.AdaptiveAvgPool2d((h, w))
        self.qkv = CrossAttention_vv(in_channels=in_channels // 2, emb_dim=32, num_heads=4)
        self.h = h
        self.w = w

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=28, w=28)
        x1_ = x1
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=28, w=28)

        x_edge_q = F.relu(self.conv_edge(x2))
        x_img_k = F.sigmoid(self.conv_img(x1))

        x_img_v = F.sigmoid(self.conv_img_v(x1_))
        x_img_v2 = F.sigmoid(self.conv_img_v2(x1_))
        x_img_v = self.Adapt_v(x_img_v)
        x_img_v2 = self.Adapt_v(x_img_v2)

        xmult, xadd = self.qkv(x_edge_q, x_img_k, x_img_v, x_img_v2)
        x_img_v = rearrange(x_img_v, "b c h w -> b (h w) c", h=self.h, w=self.w)
        x = x_img_v * xmult
        x = x + xadd

        return x

class Weighted_Feature_Fusion(nn.Module):
    def __init__(self, in_channels, num_f=4, h=8, w=8):
        super(Weighted_Feature_Fusion, self).__init__()
        self.num_f = num_f
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.in_channels, out_features=self.in_channels // 2), nn.Sigmoid(),
            nn.Linear(in_features=self.in_channels // 2, out_features=self.in_channels // 2), nn.ReLU(),
            nn.Linear(in_features=self.in_channels // 2, out_features=self.num_f)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        f_g = torch.stack(features, dim=1)
        f = sum(features)
        f = rearrange(f, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        f = self.GAP(f)
        f = f.squeeze(-1).squeeze(-1)
        weight = self.MLP(f)
        weight = self.softmax(weight)
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        f_final = weight * f_g
        f_final = torch.sum(f_final, dim=1)
        f_final = rearrange(f_final, 'b (h w) c -> b c h w', h=self.h, w=self.w)

        return f_final

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class Net(nn.Module):
    def __init__(self, batch_size=8, h=8, w=8):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = w

        self.vit1 = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.vit2 = timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output1 = SaveOutput()
        self.save_output2 = SaveOutput()

        self._register_hooks(self.vit1, self.save_output1)
        self._register_hooks(self.vit2, self.save_output2)

        self.CaFuse_1 = MaCF(in_channels=768, h=h, w=w)
        self.CaFuse_2 = MaCF(in_channels=768, h=h, w=w)
        self.CaFuse_3 = MaCF(in_channels=768, h=h, w=w)
        self.CaFuse_4 = MaCF(in_channels=768, h=h, w=w)

        self.Weighted = Weighted_Feature_Fusion(in_channels=384, h=h, w=w)

        # GAP and MLP
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=384),
            nn.Sigmoid(),
            nn.Dropout(0.02),
            nn.Linear(in_features=384, out_features=192),
            nn.GELU(),
            nn.Linear(in_features=192, out_features=1),
        )

    def _register_hooks(self, model, save_output):
        for layer in model.modules():
            if isinstance(layer, Block):
                layer.register_forward_hook(save_output)

    def extract_feature(self, save_output):
        x1 = save_output.outputs[6][:, 1:]
        x2 = save_output.outputs[8][:, 1:]
        x3 = save_output.outputs[9][:, 1:]
        x4 = save_output.outputs[11][:, 1:]
        return [x1, x2, x3, x4]

    def forward(self, x1, x2):
        """
        Feature extraction
        """
        # x1: (b, h, w)
        # x2: (b, h, w)
        x1 = x1.unsqueeze(1).float()
        x2 = x2.unsqueeze(1).float()

        _x1 = self.vit1(x1)
        x1 = self.extract_feature(self.save_output1)
        self.save_output1.clear()

        _x2 = self.vit2(x2)
        x2 = self.extract_feature(self.save_output2)
        self.save_output2.clear()

        # f_img: [(b, 784, 768), (b, 784, 768), (b, 784, 768), (b, 784, 768),]
        # f_imge: [(b, 784, 768), (b, 784, 768), (b, 784, 768), (b, 784, 768),]

        f_img = x1
        f_imge = x2

        """
        Multi-scale adaptive cross feature fusion
        """
        f = []
        for i in range(4):
            tmp = getattr(self, f'CaFuse_{i+1}')(f_imge[i], f_img[i])
            f.append(tmp)

        # f: [(b, 64, 384), (b, 64, 384), (b, 64, 384), (b, 64, 384)]

        """
        Weighted feature fusion and quality prediction
        """

        out = self.Weighted(f)
        out = self.score(out)
        out = out.squeeze(-1)

        return out.squeeze(-1)

if __name__ == "__main__":
    x1 = torch.randn(4, 224, 224)
    x2 = torch.randn(4, 224, 224)
    net = Net(batch_size=4)

    out = net(x1, x2)
    print(out.shape)
