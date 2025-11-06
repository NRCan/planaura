from planaura.networks.backbones.planaura_reconstruction import planaura_reconstruction


def generate_backbone(config):
    backbone = config["model_params"]["backbone"]
    if backbone == 'planaura_reconstruction':
        return planaura_reconstruction(config)
    else:
        raise ValueError('Architecture {} is not recognized!'.format(backbone))
