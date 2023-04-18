import torch
from .layers import FourierConv1d, FourierConv2d


class FNO_GRU_1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, width):
        super(FNO_GRU_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.z_layer = FourierConv1d(self.width, self.width, spatial_size)
        self.r_layer = FourierConv1d(self.width, self.width, spatial_size)
        self.h_layer = FourierConv1d(self.width, self.width, spatial_size)

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        x = self.in_mapping(x)
        output = torch.zeros(
            x.shape[0], num_time_steps, x.shape[1], self.out_channels
        ).to(x.device)
        for i in range(num_time_steps):
            x = self.cell(x)
            output[:, i, :, :] = self.out_mapping(x)
        return output

    def cell(self, h):
        h_in = h.permute(0, 2, 1)
        z = torch.sigmoid(self.z_layer(h_in))
        r = torch.sigmoid(self.r_layer(h_in))
        new_h = torch.tanh(self.h_layer(r * h_in))
        h = (1 - z) * h_in + z * new_h
        return h.permute(0, 2, 1)


class FNO_GRU_2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, spatial_size_x, spatial_size_y, width
    ):
        super(FNO_GRU_2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.in_mapping = torch.nn.Linear(in_channels, self.width)
        # print(f"instantiate z_layer")
        self.z_layer = FourierConv2d(
            self.width, self.width, spatial_size_x, spatial_size_y
        )
        # print(f"instantiate r_layer")
        self.r_layer = FourierConv2d(
            self.width, self.width, spatial_size_x, spatial_size_y
        )
        # print(f"instantiate h_layer")
        self.h_layer = FourierConv2d(
            self.width, self.width, spatial_size_x, spatial_size_y
        )

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        # print(f"Shape of x before in_mapping: {x.shape}")
        x = self.in_mapping(x)
        # print(f"Shape of x after in_mapping: {x.shape}")
        output = torch.zeros(
            x.shape[0], num_time_steps, x.shape[1], x.shape[2], self.out_channels
        ).to(x.device)
        for i in range(num_time_steps):
            # print(f"Applying FNO_GRU_2d cell time step {i} to x.shape: {x.shape}")
            x = self.cell(x)
            # print(f"x.shape after cell: {x.shape}")
            output[:, i, :, :, :] = self.out_mapping(x)
        return output

    def cell(self, h):
        # print(f"Shape of h, start of cell: {h.shape}")
        h_in = h.permute(0, 3, 1, 2)
        # print(f"Shape of h_in: {h_in.shape}")
        z = torch.sigmoid(self.z_layer(h_in))
        # print(f"Shape of z: {z.shape}")
        r = torch.sigmoid(self.r_layer(h_in))
        # print(f"Shape of r: {r.shape}")
        new_h = torch.tanh(self.h_layer(r * h_in))
        # print(f"Shape of new_h: {new_h.shape}")
        h = (1 - z) * h_in + z * new_h
        return h.permute(0, 2, 3, 1)
