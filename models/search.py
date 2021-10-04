import torch.nn as nn
from enot.models import SearchVariantsContainer

from models.common import C3
from models.common import Conv
from models.common import ConvBlock


class ConvC3(nn.Module):
    # Convolution layer followed by CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, C3 kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=0.5, e0=1.0, k_c3=3):
        super().__init__()
        c_ = int(c2 * e0)  # hidden channels
        self.cv = Conv(c1, c_, k=k, s=s)
        # e * e0 - to make bottleneck channels be equal to round(c2 * e * e0) so it changes with the e0 value
        self.c3 = C3(c_, c2, n=n, shortcut=shortcut, g=g, e=(e * e0), k=k_c3)

    def forward(self, x):
        return self.c3(self.cv(x))


class C3Conv(nn.Module):
    # CSP Bottleneck with 3 convolutions followed by convolution layer
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, C3 kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=0.5, e0=1.0, k_c3=3):
        super().__init__()
        c_ = int(c2 * e0)  # hidden channels
        # here again the number of bottleneck channels is equal to round(c2 * e * e0)
        self.c3 = C3(c1, c_, n=n, shortcut=shortcut, g=g, e=e, k=k_c3)
        self.cv = Conv(c_, c2, k=k, s=s)

    def forward(self, x):
        return self.cv(self.c3(x))


class ReductionConvBlock(nn.Module):
    # Convolution layer followed by convolution block with skip connection
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, conv kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=1.0, e0=0.5, k_block=3):
        super().__init__()
        c_ = int(c2 * e0)  # hidden channels
        self.cv = Conv(c1, c_, k=k, s=s)
        # e * e0 - to make bottleneck channels be equal to round(c2 * e * e0) so it changes with the e0 value
        self.conv_block = ConvBlock(c_, c2, n=n, k=k_block, shortcut=shortcut, e=(e * e0))

    def forward(self, x):
        return self.conv_block(self.cv(x))


class C3Block(nn.Module):
    # CSP Bottleneck Search Block with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion, kernel
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):

        super().__init__()

        search_variants = []

        combinations = [
            (0.50, 1.00, 3),  # 0
            (0.50, 0.66, 3),  # 1
            (0.50, 0.33, 3),  # 2
            (0.25, 1.00, 3),  # 3
            (0.25, 0.66, 3),  # 4
            (0.25, 0.33, 3),  # 5
            (0.25, 0.66, 5),  # 6
            (0.25, 0.33, 5),  # 7
        ]
        for e, d, k in combinations:
            d = max(int(round(n * d)), 1)
            search_variants.append(C3(
                c1, c2, n=d, shortcut=shortcut, g=g, e=e, k=k,
            ))

        # kernels = [3, 5]
        # expansions = [0.5, 0.25]
        # depths = [1.0, 0.66, 0.33]
        # depths = [max(int(round(n * d)), 1) for d in depths]
        # for e in expansions:
        #     for d in depths:
        #         for k in kernels:
        #             search_variants.append(C3(c1, c2, n=d, shortcut=shortcut, g=g, e=e, k=k))

        search_block = SearchVariantsContainer(search_variants, default_operation_index=0)
        self.search_block = search_block

    def forward(self, x):
        return self.search_block(x)


class SearchBlockSmall(nn.Module):
    # Small search block for yolov5s
    # ch_in, ch_out, number, stride, shortcut, groups, expansion, kernel
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):

        super().__init__()

        search_variants = []

        combinations = [
        #    type     e      d     k
            ('c3',    0.500, 1.00, 3),
            ('c3',    0.375, 1.00, 3),
            ('c3',    0.250, 1.00, 3),

            ('convs', 0.500, 2.00, 3),
            ('convs', 0.375, 2.00, 3),
            ('convs', 0.250, 2.00, 3),
            ('convs', 0.125, 2.00, 3),

            ('convs', 0.375, 3.00, 3),
            ('convs', 0.250, 3.00, 3),
            ('convs', 0.125, 3.00, 3),

            ('convs', 0.250, 4.00, 3),
            ('convs', 0.125, 4.00, 3),
        ]

        for block_type, e, d, k in combinations:
            d = max(int(round(n * d)), 1)
            if block_type == 'c3':
                search_variants.append(C3(
                    c1, c2, n=d, shortcut=shortcut, g=g, e=e, k=k,
                ))
            else:
                search_variants.append(ConvBlock(
                    c1, c2, n=d, k=k, shortcut=shortcut, e=e,
                ))

        search_block = SearchVariantsContainer(search_variants, default_operation_index=0)
        self.search_block = search_block

    def forward(self, x):
        return self.search_block(x)


class ConvC3Block(nn.Module):
    # Encoder v2 search block
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, C3 kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=0.5, e0=1.0, k_c3=3):

        super().__init__()

        search_variants = []

        combinations = [
            (1.0, 1.00, 3),  # 0
            (1.0, 0.66, 3),  # 1
            (1.0, 0.33, 3),  # 2
            (0.5, 1.00, 3),  # 3
            (0.5, 0.66, 3),  # 4
            (0.5, 0.33, 3),  # 5
            (0.5, 0.66, 5),  # 6
            (0.5, 0.33, 5),  # 7
        ]
        for e0, d, k in combinations:
            d = max(int(round(n * d)), 1)
            search_variants.append(ConvC3(
                c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_c3=k,
            ))

        # kernels = [3, 5]
        # expansions = [1.0, 0.5]
        # depths = [1.0, 0.66, 0.33]
        # depths = [max(int(round(n * d)), 1) for d in depths]
        # for k in kernels:
        #     for e0 in expansions:
        #         for d in depths:
        #             search_variants.append(
        #                 ConvC3(c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_c3=k)
        #             )

        search_block = SearchVariantsContainer(search_variants, default_operation_index=0)
        self.search_block = search_block

    def forward(self, x):
        return self.search_block(x)


class ReductionSearchBlockSmall(nn.Module):
    # Small search block for yolov5s with spatial reduction
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, block kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=0.5, e0=1.0, k_c3=3):

        super().__init__()

        search_variants = []

        if n == 1:
            combinations = [
            #    type     e0    e     d     k
                ('c3',    1.00, 0.50, 1.00, 3),
                ('c3',    0.75, 0.75, 1.00, 3),
                ('c3',    0.50, 1.00, 1.00, 3),

                ('convs', 1.00, 0.50, 1.00, 3),
                ('convs', 0.75, 0.50, 1.00, 3),
                ('convs', 0.50, 0.50, 1.00, 3),

                ('convs', 0.75, 0.50, 2.00, 3),
                ('convs', 0.50, 0.75, 2.00, 3),
                ('convs', 0.50, 0.50, 2.00, 3),

                ('convs', 0.75, 0.375, 3.00, 3),
                ('convs', 0.50, 0.500, 3.00, 3),
                ('convs', 0.50, 0.375, 3.00, 3),
            ]
        else:
            combinations = [
            #    type     e0    e     d     k
                ('c3',    1.00, 0.50, 1.00, 3),
                ('c3',    0.75, 0.75, 1.00, 3),
                ('c3',    0.75, 0.50, 1.00, 3),
                ('c3',    0.50, 1.00, 1.00, 3),
                ('c3',    0.50, 0.75, 1.00, 3),
                ('c3',    0.50, 0.50, 1.00, 3),

                ('c3',    1.00, 0.50, 0.66, 3),
                ('c3',    0.75, 0.75, 0.66, 3),
                ('c3',    0.75, 0.50, 0.66, 3),
                ('c3',    0.50, 1.00, 0.66, 3),
                ('c3',    0.50, 0.75, 0.66, 3),
                ('c3',    0.50, 0.50, 0.66, 3),
            ]

        for block_type, e0, e, d, k in combinations:
            d = max(int(round(n * d)), 1)
            if block_type == 'c3':
                search_variants.append(ConvC3(
                    c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_c3=k,
                ))
            else:
                search_variants.append(ReductionConvBlock(
                    c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_block=k,
                ))

        search_block = SearchVariantsContainer(search_variants, default_operation_index=0)
        self.search_block = search_block

    def forward(self, x):
        return self.search_block(x)


class C3ConvBlock(nn.Module):
    # Decoder v2 search block
    # ch_in, ch_out, number, kernel, stride, shortcut, groups, expansion, first conv expansion, C3 kernel
    def __init__(self, c1, c2, n=1, k=3, s=1, shortcut=True, g=1, e=0.5, e0=1.0, k_c3=3):

        super().__init__()

        search_variants = []

        combinations = [
            (1.0, 1.00, 3),  # 0
            (1.0, 0.66, 3),  # 1
            (1.0, 0.33, 3),  # 2
            (0.5, 1.00, 3),  # 3
            (0.5, 0.66, 3),  # 4
            (0.5, 0.33, 3),  # 5
            (0.5, 0.66, 5),  # 6
            (0.5, 0.33, 5),  # 7
        ]
        for e0, d, k in combinations:
            d = max(int(round(n * d)), 1)
            search_variants.append(C3Conv(
                c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_c3=k,
            ))

        # kernels = [3, 5]
        # expansions = [1.0, 0.5]
        # depths = [1.0, 0.66, 0.33]
        # depths = [max(int(round(n * d)), 1) for d in depths]
        # for k in kernels:
        #     for e0 in expansions:
        #         for d in depths:
        #             search_variants.append(
        #                 C3Conv(c1, c2, n=d, k=k, s=s, shortcut=shortcut, g=g, e=e, e0=e0, k_c3=k)
        #             )

        search_block = SearchVariantsContainer(search_variants, default_operation_index=0)
        self.search_block = search_block

    def forward(self, x):
        return self.search_block(x)
