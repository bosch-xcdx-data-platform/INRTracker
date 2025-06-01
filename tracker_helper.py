from src.generator.pretrained_generator import trainedGET3D_G

def get_GET3D_model(G_kwargs, common_kwargs, generator_checkpoint):
    return trainedGET3D_G(G_kwargs, common_kwargs=common_kwargs, checkpoint_path=generator_checkpoint), None

def get_object_model(generator_checkpoint,  G_kwargs, common_kwargs, n_trained_models=None, **kwargs):
    if G_kwargs["name"] == "GET3D":
        return get_GET3D_model(G_kwargs, common_kwargs=common_kwargs, generator_checkpoint=generator_checkpoint)       
    else:
        return None