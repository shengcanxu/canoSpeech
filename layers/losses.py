
import torch
from coqpit import Coqpit
from torch import nn

from layers.soft_dtw import SoftDTW
from util.helper import sequence_mask
from torch.nn import functional as F

class VitsGeneratorLoss(nn.Module):
    def __init__(self, config: Coqpit):
        super().__init__()
        self.kl_loss_alpha = config.loss.kl_loss_alpha
        self.gen_loss_alpha = config.loss.gen_loss_alpha
        self.feat_loss_alpha = config.loss.feat_loss_alpha
        self.dur_loss_alpha = config.loss.dur_loss_alpha
        self.mel_loss_alpha = config.loss.mel_loss_alpha
        self.spk_encoder_loss_alpha = config.loss.speaker_encoder_loss_alpha

    @staticmethod
    def feature_loss(feats_real, feats_generated):
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        # kl_loss 的介绍在视频 https://www.bilibili.com/video/BV1VG411h75N/?spm_id_from=333.788&vd_source=d38c9d5cf896f215d746bb79474d6606
        # 的 1:38:43 开始处
        # 对于kl_loss 通过mean 和standard deviation计算的方法来计算两个高斯分布的KL散度， 详细解释看：
        # https://zhuanlan.zhihu.com/p/345095899
        # 或者 https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)

        # don't know why kl_loss will be negative, maybe it's because the batch size is not big enough that makes the sample numbers not big enough to make it ocationaly be negative.
        l = torch.abs(l)
        return l

    @staticmethod
    def cosine_similarity_loss(gt_spk_emb, syn_spk_emb):
        return -torch.nn.functional.cosine_similarity(gt_spk_emb, syn_spk_emb).mean()

    def forward(
        self,
        mel_slice,  # [B, 1, T]
        mel_slice_hat,  # [B, 1, T]
        z_p,  # [B, C, T]
        logs_q,  # [B, C, T]
        m_p,  # [B, C, T]
        logs_p,  # [B, C, T]
        z_len,  # [B]
        scores_disc_fake,  # [B, C]
        feats_disc_fake,  # [B, C, T', P]
        feats_disc_real,  # [B, C, T', P]
        loss_duration,
        use_speaker_encoder_as_loss=False,
        gt_spk_emb=None,
        syn_spk_emb=None,
    ):
        loss = 0.0
        return_dict = {}
        z_mask = sequence_mask(z_len).float()
        # compute losses
        loss_kl = (
            self.kl_loss(z_p=z_p, logs_q=logs_q, m_p=m_p, logs_p=logs_p, z_mask=z_mask.unsqueeze(1))
            * self.kl_loss_alpha
        )
        loss_feat = (
            self.feature_loss(feats_real=feats_disc_real, feats_generated=feats_disc_fake) * self.feat_loss_alpha
        )
        loss_gen = self.generator_loss(scores_fake=scores_disc_fake)[0] * self.gen_loss_alpha
        loss_mel = torch.nn.functional.l1_loss(mel_slice, mel_slice_hat) * self.mel_loss_alpha
        loss_duration = torch.sum(loss_duration.float()) * self.dur_loss_alpha
        loss = loss_kl + loss_feat + loss_mel + loss_gen + loss_duration

        if use_speaker_encoder_as_loss:
            loss_se = self.cosine_similarity_loss(gt_spk_emb, syn_spk_emb) * self.spk_encoder_loss_alpha
            loss = loss + loss_se
            return_dict["loss_spk_encoder"] = loss_se
        # pass losses to the dict
        return_dict["loss_gen"] = loss_gen
        return_dict["loss_kl"] = loss_kl
        return_dict["loss_feat"] = loss_feat
        return_dict["loss_mel"] = loss_mel
        return_dict["loss_duration"] = loss_duration
        return_dict["loss"] = loss
        return return_dict


class VitsDiscriminatorLoss(nn.Module):
    def __init__(self, config: Coqpit):
        super().__init__()
        self.disc_loss_alpha = config.loss.disc_loss_alpha

    @staticmethod
    def discriminator_loss(scores_real, scores_fake):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(scores_real, scores_fake):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1 - dr) ** 2)
            fake_loss = torch.mean(dg**2)
            loss += real_loss + fake_loss
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())
        return loss, real_losses, fake_losses

    def forward(self, scores_disc_real, scores_disc_fake):
        loss = 0.0
        return_dict = {}
        loss_disc, loss_disc_real, _ = self.discriminator_loss(
            scores_real=scores_disc_real,
            scores_fake=scores_disc_fake
        )
        return_dict["loss_disc"] = loss_disc * self.disc_loss_alpha
        loss = loss + return_dict["loss_disc"]
        return_dict["loss"] = loss

        for i, ldr in enumerate(loss_disc_real):
            return_dict[f"loss_disc_real_{i}"] = ldr
        return return_dict

sdtw = SoftDTW(use_cuda=False, gamma=0.01, warp=134.4)
class NaturalSpeechGeneratorLoss(nn.Module):
    def __init__(self, config: Coqpit):
        super().__init__()
        self.kl_loss_alpha = config.loss.kl_loss_alpha
        self.kl_loss_forward_alpha = config.loss.kl_loss_forward_alpha
        self.gen_loss_alpha = config.loss.gen_loss_alpha
        self.gen_loss_e2e_alpha = config.loss.gen_loss_e2e_alpha
        self.feat_loss_alpha = config.loss.feat_loss_alpha
        self.dur_loss_alpha = config.loss.dur_loss_alpha
        self.pitch_loss_alpha = config.loss.pitch_loss_alpha
        self.mel_loss_alpha = config.loss.mel_loss_alpha
        self.spk_encoder_loss_alpha = config.loss.speaker_encoder_loss_alpha

        self.use_soft_dynamic_time_warping = config.loss.use_soft_dynamic_time_warping

    @staticmethod
    def feature_loss(feats_real, feats_generated):
        """total loss range: [0, (8+6*5)*2=76]"""
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):  # list of 6 items
            for rl, gl in zip(dr, dg): # 8 items for scale disc, 6 items for those 5 period disc
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))  # range [0, 1]
        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        """loss range [0, 6], because each loss range [0, 1]"""
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        # kl_loss 的介绍在视频 https://www.bilibili.com/video/BV1VG411h75N/?spm_id_from=333.788&vd_source=d38c9d5cf896f215d746bb79474d6606
        # 的 1:38:43 开始处
        # 对于kl_loss 通过mean 和standard deviation计算的方法来计算两个高斯分布的KL散度， 详细解释看：
        # https://zhuanlan.zhihu.com/p/345095899
        # 或者 https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)

        # don't know why kl_loss will be negative, maybe it's because the batch size is not big enough that makes the sample numbers not big enough to make it ocationaly be negative.
        l = torch.abs(l)
        return l

    @staticmethod
    def kl_loss_sdtw(z_p, logs_q, m_p, logs_p, p_mask, q_mask):
        INF = 1e5

        kl = NaturalSpeechGeneratorLoss.get_sdtw_kl_matrix(z_p, logs_q, m_p, logs_p)  # [b t_tp t_tq]
        kl = torch.nn.functional.pad(kl, (0, 1, 0, 1), "constant", 0)
        p_mask = torch.nn.functional.pad(p_mask, (0, 1), "constant", 0)
        q_mask = torch.nn.functional.pad(q_mask, (0, 1), "constant", 0)

        kl.masked_fill_(p_mask[:, :, None].bool() ^ q_mask[:, None, :].bool(), INF)
        kl.masked_fill_((~p_mask[:, :, None].bool()) & (~q_mask[:, None, :].bool()), 0)
        res = sdtw(kl).sum() / p_mask.sum()
        return res

    @staticmethod
    def get_sdtw_kl_matrix(z_p, logs_q, m_p, logs_p):
        """
        returns kl matrix with shape [b, t_tp, t_tq]
        z_p, logs_q: [b, h, t_tq]
        m_p, logs_p: [b, h, t_tp]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()

        t_tp, t_tq = m_p.size(-1), z_p.size(-1)
        b, h, t_tp = m_p.shape

        kls = torch.zeros((b, t_tp, t_tq), dtype=z_p.dtype, device=z_p.device)
        for i in range(h):
            logs_p_, m_p_, logs_q_, z_p_ = (logs_p[:, i, :, None], m_p[:, i, :, None], logs_q[:, i, None, :], z_p[:, i, None, :],)
            kl = logs_p_ - logs_q_ - 0.5  # [b, t_tp, t_tq]
            kl += 0.5 * ((z_p_ - m_p_) ** 2) * torch.exp(-2.0 * logs_p_)
            kls += kl
        return kls

        kl = logs_p[:, :, :, None] - logs_q[:, :, None, :] - 0.5  # p, q
        kl += (0.5 * ((z_p[:, :, None, :] - m_p[:, :, :, None]) ** 2) * torch.exp(-2.0 * logs_p[:, :, :, None]))

        kl = kl.sum(dim=1)
        return kl

    @staticmethod
    def cosine_similarity_loss(gt_spk_emb, syn_spk_emb):
        return -torch.nn.functional.cosine_similarity(gt_spk_emb, syn_spk_emb).mean()

    def forward(
        self,
        mel_slice,  # [B, 1, T]
        mel_slice_hat,  # [B, 1, T]
        scores_disc_fake,
        scores_disc_fake_e2e,
        feats_disc_real,
        feats_disc_fake,
        duration_loss,
        pitch_loss,
        z_p,
        m_p,
        logs_p,
        z_q,
        m_q,
        logs_q,
        p_mask,
        z_mask
    ):
        loss_gen, losses_gen = self.generator_loss(scores_disc_fake)  # range [0, 6]
        loss_gen = loss_gen * self.gen_loss_alpha
        loss_gen_e2e, losses_gen_e2e = self.generator_loss(scores_disc_fake_e2e)  # range [0, 6]
        loss_gen_e2e = loss_gen_e2e * self.gen_loss_e2e_alpha

        # feature loss of discriminator, range: [0, (8+6*5)*2=76]
        loss_fm = self.feature_loss(feats_disc_real, feats_disc_fake) * self.feat_loss_alpha
        # mel loss, range: [0, inf], normally in [0, 20]
        loss_mel = F.l1_loss(mel_slice, mel_slice_hat) * self.mel_loss_alpha

        # duration and pitch loss generated from text
        loss_dur = torch.sum(duration_loss.float()) * self.dur_loss_alpha
        loss_pitch = torch.sum(pitch_loss.float()) * self.pitch_loss_alpha

        # kl loss makes z generated from audio and z_q generated from text are in the same distribution
        if self.use_soft_dynamic_time_warping:
            loss_kl = self.kl_loss_sdtw(z_p, logs_q, m_p, logs_p, p_mask, z_mask) * self.kl_loss_alpha
            loss_kl_fwd = self.kl_loss_sdtw(z_q, logs_p, m_q, logs_q, z_mask, p_mask) * self.kl_loss_forward_alpha
        else:
            loss_kl = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.kl_loss_alpha
            loss_kl_fwd = self.kl_loss(z_q, logs_p, m_q, logs_q, p_mask) * self.kl_loss_forward_alpha

        # total loss = sum all losses
        loss = loss_gen + loss_gen_e2e + loss_fm + loss_mel + loss_dur + loss_pitch + loss_kl + loss_kl_fwd

        return_dict = {}
        # pass losses to the dict
        return_dict["loss_gen"] = loss_gen
        return_dict["loss_gen_e2e"] = loss_gen_e2e
        return_dict["loss_feature"] = loss_fm
        return_dict["loss_mel"] = loss_mel
        return_dict["loss_duration"] = loss_dur
        return_dict["loss_pitch"] = loss_pitch
        return_dict["loss_kl"] = loss_kl
        return_dict["loss_kl_forward"] = loss_kl_fwd
        return_dict["loss"] = loss
        return return_dict


class NaturalSpeechDiscriminatorLoss(nn.Module):
    """the same with VitsDiscriminatorLost"""
    def __init__(self, config: Coqpit):
        super().__init__()
        self.disc_loss_alpha = config.loss.disc_loss_alpha

    @staticmethod
    def discriminator_loss(scores_real, scores_fake):
        """the input data size doesnot affect the loss range. each real_loss/fake_loss range: [0,1],
            so loss range [0, 2*6], *6 because there are 5 period disc and 1 scale disc """
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(scores_real, scores_fake):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1 - dr) ** 2)
            fake_loss = torch.mean(dg**2)
            loss += real_loss + fake_loss
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())
        return loss, real_losses, fake_losses

    def forward(self, scores_disc_real, scores_disc_fake):
        loss = 0.0
        return_dict = {}
        loss_disc, loss_disc_real, loss_disc_fake = self.discriminator_loss(  # range [0, 2*6]
            scores_real=scores_disc_real,
            scores_fake=scores_disc_fake
        )
        return_dict["loss_disc"] = loss_disc * self.disc_loss_alpha
        loss = loss + return_dict["loss_disc"]
        return_dict["loss"] = loss

        return_dict["loss_disc_real_all"] = sum(loss_disc_real) / len(loss_disc_real) * self.disc_loss_alpha
        return_dict["loss_disc_fake_all"] = sum(loss_disc_fake) / len(loss_disc_fake) * self.disc_loss_alpha
        return return_dict

