from torch import nn
from layers.hifigan_discriminator import DiscriminatorPeriodic, DiscriminatorScale


class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.
    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^
    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """
    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorScale(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorPeriodic(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x, x_hat=None):
        """Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.
        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats
