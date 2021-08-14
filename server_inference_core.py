"""
Heart of most evaluation scripts (DAVIS semi-sup/interactive, GUI)
Handles propagation and fusion
See eval_semi_davis.py / eval_interactive_davis.py for examples
"""

import torch
import numpy as np

from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.aggregate import aggregate_wbg

from util.tensor_util import pad_divide_by


class InferenceCore:
    """
    images - leave them in original dimension (unpadded), but do normalize them. 
            Should be CPU tensors of shape B*T*3*H*W
            
    mem_profile - How extravagant I can use the GPU memory. 
                Usually more memory -> faster speed but I have not drawn the exact relation
                0 - Use the most memory
                1 - Intermediate, larger buffer 
                2 - Intermediate, small buffer 
                3 - Use the minimal amount of GPU memory
                Note that *none* of the above options will affect the accuracy
                This is a space-time tradeoff, not a space-performance one

    mem_freq - Period at which new memory are put in the bank
                Higher number -> less memory usage
                Unlike the last option, this *is* a space-performance tradeoff
    """

    def __init__(self, prop_net: PropagationNetwork, fuse_net: FusionNet, images, mem_freq=1, device='cuda:0', num_objects=1):
        self.prop_net = prop_net.to(device, non_blocking=True)
        if fuse_net is not None:
            self.fuse_net = fuse_net.to(device, non_blocking=True)
        self.mem_freq = mem_freq
        self.device = device

        self.data_dev = device
        self.result_dev = device
        self.k_buf_size = 105
        self.i_buf_size = -1  # no need to buffer image

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]
        self.k = num_objects

        # Pad each side to multiples of 16
        self.images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        # Padded dimensions
        nh, nw = self.images.shape[-2:]
        self.images = self.images.to(self.data_dev, non_blocking=False)

        # TODO: load prob from stored masks if fusion
        # Object probabilities, background included
        self.prob = torch.zeros((self.k, t, 1, nh, nw),
                                dtype=torch.float32, device=self.result_dev)

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.key_buf = {}
        self.image_buf = {}

        self.certain_mem_k = None
        self.certain_mem_v = None

    def get_image_buffered(self, idx):
        if self.data_dev == self.device:
            return self.images[:, idx]

        # buffer the .cuda() calls
        if idx not in self.image_buf:
            # Flush buffer
            if len(self.image_buf) > self.i_buf_size:
                self.image_buf = {}
        self.image_buf[idx] = self.images[:, idx].to(self.device)
        result = self.image_buf[idx]

        return result

    def get_key_feat_buffered(self, idx):
        if idx not in self.key_buf:
            # Flush buffer
            if len(self.key_buf) > self.k_buf_size:
                self.key_buf = {}

            self.key_buf[idx] = self.prop_net.encode_key(
                self.get_image_buffered(idx))
        result = self.key_buf[idx]

        return result

    def do_pass(self, key_k, key_v, idx):
        """
        Do a complete pass that includes propagation and fusion
        key_k/key_v -  memory feature of the starting frame
        idx - Frame index of the starting frame
        """

        # Pointer in the memory bank
        num_certain_keys = self.certain_mem_k.shape[2]
        m_front = num_certain_keys

        # Determine the required size of the memory bank
        closest_ti = self.t
        total_m = (closest_ti - idx - 1)//self.mem_freq + \
            1 + num_certain_keys

        _, CK, _, H, W = key_k.shape
        K, CV, _, _, _ = key_v.shape

        # Pre-allocate keys/values memory
        keys = torch.empty((1, CK, total_m, H, W),
                           dtype=torch.float32, device=self.device)
        values = torch.empty((K, CV, total_m, H, W),
                             dtype=torch.float32, device=self.device)

        # Initial key/value passed in
        keys[:, :, 0:num_certain_keys] = self.certain_mem_k
        values[:, :, 0:num_certain_keys] = self.certain_mem_v
        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range:
            this_k = keys[:, :, :m_front]
            this_v = values[:, :, :m_front]
            k16, qv16, qf16, qf8, qf4 = self.get_key_feat_buffered(ti)
            out_mask = self.prop_net.segment_with_query(
                this_k, this_v, qf8, qf4, k16, qv16)

            out_mask = aggregate_wbg(out_mask, keep_bg=False)

            if ti != end and abs(ti-last_ti) >= self.mem_freq:
                keys[:, :, m_front:m_front+1] = k16.unsqueeze(2)
                values[:, :, m_front:m_front+1] = self.prop_net.encode_value(
                    self.get_image_buffered(ti), qf16, out_mask[1:])

                m_front += 1
                last_ti = ti

            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            if (closest_ti != self.t) and (closest_ti != -1):
                self.prob[:, ti] = self.fuse_one_frame(closest_ti, idx, ti, self.prob[:, ti], out_mask,
                                                       key_k, k16).to(self.result_dev, non_blocking=True)
            else:
                self.prob[:, ti] = out_mask.to(
                    self.result_dev, non_blocking=True)

        return closest_ti

    def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask, mk16, qk16):
        assert(tc < ti < tr or tr < ti < tc)

        prob = torch.zeros((self.k, 1, self.nh, self.nw),
                           dtype=torch.float32, device=self.device)

        # Compute linear coefficients
        nc = abs(tc-ti) / abs(tc-tr)
        nr = abs(tr-ti) / abs(tc-tr)
        dist = torch.FloatTensor([nc, nr]).to(self.device).unsqueeze(0)
        attn_map = self.prop_net.get_attention(
            mk16, self.pos_mask_diff, self.neg_mask_diff, qk16)
        for k in range(0, self.k):
            w = torch.sigmoid(self.fuse_net(self.get_image_buffered(ti),
                                            prev_mask[k:k+1].to(self.device), curr_mask[k:k+1].to(self.device), attn_map[k:k+1], dist))
            prob[k] = w
        return aggregate_wbg(prob, keep_bg=False)

    def interact(self, mask, idx):
        """
        Interact -> Propagate -> Fuse

        mask - One-hot mask of the interacted frame, background included
        idx - Frame index of the interacted frame

        Return: all mask prob images
        """

        mask = mask.to(self.device)
        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        self.mask_diff = mask - self.prob[:, idx].to(self.device)
        self.pos_mask_diff = self.mask_diff.clamp(0, 1)
        self.neg_mask_diff = (-self.mask_diff).clamp(0, 1)

        self.prob[:, idx] = mask
        key_k, _, qf16, _, _ = self.get_key_feat_buffered(idx)
        key_k = key_k.unsqueeze(2)
        key_v = self.prop_net.encode_value(
            self.get_image_buffered(idx), qf16, mask)

        if self.certain_mem_k is None:
            self.certain_mem_k = key_k
            self.certain_mem_v = key_v
        else:
            self.certain_mem_k = torch.cat([self.certain_mem_k, key_k], 2)
            self.certain_mem_v = torch.cat([self.certain_mem_v, key_v], 2)

        # Forward
        self.do_pass(key_k, key_v, idx, True)

        # T * H * W
        result_prob = self.prob[0]
        # Trim paddings
        if self.pad[2]+self.pad[3] > 0:
            result_prob = result_prob[:, :, self.pad[2]:-self.pad[3], :]
        if self.pad[0]+self.pad[1] > 0:
            result_prob = result_prob[:, :, :, self.pad[0]:-self.pad[1]]

        prob_numpy = (result_prob.detach().cpu().numpy()[:, 0] * 255).astype(np.uint8)

        return prob_numpy
