"""Read an INT4 semantic-page record before its transient MLA projection."""

import triton
import triton.language as tl


@triton.jit
def load_latent(source, scales, sums, sum_scales, page_counts,
                batch, page, leaf, channel, valid,
                TOKENS: tl.constexpr, PAGES: tl.constexpr, WIDTH: tl.constexpr,
                GROUP: tl.constexpr, INT8_SUMS: tl.constexpr):
    """Kimi/GLM have one physical KV head and complete residual-INT4 pages."""
    packed = tl.load(source + (batch * TOKENS + leaf[:, None]) * (WIDTH // 2)
                     + channel[None, :] // 2, mask=valid[:, None], other=0).to(tl.int32)
    code = ((packed >> ((channel[None, :] & 1) * 4)) & 15) - 8
    scale = tl.load(scales + (batch * PAGES + page[:, None]) * (WIDTH // GROUP)
                    + channel[None, :] // GROUP, mask=valid[:, None], other=0).to(tl.float32)
    anchor = tl.load(sums + (batch * PAGES + page[:, None]) * WIDTH + channel[None, :],
                     mask=valid[:, None], other=0).to(tl.float32)
    if INT8_SUMS:
        anchor *= tl.load(sum_scales + (batch * PAGES + page[:, None]) * (WIDTH // GROUP)
                           + channel[None, :] // GROUP, mask=valid[:, None], other=0).to(tl.float32)
    count = tl.load(page_counts + batch * PAGES + page, mask=valid, other=1).to(tl.float32)
    return (code * scale + anchor / tl.maximum(count[:, None], 1.)).to(tl.bfloat16)
