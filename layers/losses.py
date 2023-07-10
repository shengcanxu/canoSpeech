
import torch
from coqpit import Coqpit
from torch import nn
from util.helper import sequence_mask


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

        # kl_loss 的介绍在视频 https://www.bilibili.com/video/BV1VG411h75N/?spm_id_from=333.788&vd_source=d38c9d5cf896f215d746bb79474d6606 的 1:38:43 开始处
        # 对于kl_loss 通过mean 和standard deviation计算的方法来计算两个高斯分布的KL散度， 详细解释看：https://zhuanlan.zhihu.com/p/345095899 或者 https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l

    @staticmethod
    def cosine_similarity_loss(gt_spk_emb, syn_spk_emb):
        return -torch.nn.functional.cosine_similarity(gt_spk_emb, syn_spk_emb).mean()

    def forward(
        self,
        mel_slice,
        mel_slice_hat,
        z_p,
        logs_q,
        m_p,
        logs_p,
        z_len,
        scores_disc_fake,
        feats_disc_fake,
        feats_disc_real,
        loss_duration,
        use_speaker_encoder_as_loss=False,
        gt_spk_emb=None,
        syn_spk_emb=None,
    ):
        """
        Shapes:
            - mel_slice : :math:`[B, 1, T]`
            - mel_slice_hat: :math:`[B, 1, T]`
            - z_p: :math:`[B, C, T]`
            - logs_q: :math:`[B, C, T]`
            - m_p: :math:`[B, C, T]`
            - logs_p: :math:`[B, C, T]`
            - z_len: :math:`[B]`
            - scores_disc_fake[i]: :math:`[B, C]`
            - feats_disc_fake[i][j]: :math:`[B, C, T', P]`
            - feats_disc_real[i][j]: :math:`[B, C, T', P]`
        """
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

