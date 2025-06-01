# pretrained_generator.py

import torch
import submodules.GET3D.dnnlib as dnnlib
import copy

class trainedGET3D_G:
    def __init__(self, 
                 G_kwargs, 
                 common_kwargs,
                 checkpoint_path):
        
        # Load the pretrained model from the checkpoint_path
        device = G_kwargs['device']
        
        # G, G_ema class: GeneratorDMTETMesh
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
            device)  # subclass of torch.nn.Module
        G_ema = copy.deepcopy(G).eval().to(device)  # deepcopy can make sure they are correct.
        
        print('Loading pretrained model from checkpoint path: {}'.format(checkpoint_path))
        model_state_dict = torch.load(checkpoint_path, map_location=device)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        
        self.mapping_tex = G_ema.mapping # class: MappingNetwork
        self.mapping_geo = G_ema.mapping_geo # class: MappingNetwork
        self.generator = G_ema.synthesis # class: DMTETSynthesisNetwork
        self.renderer = self.generator.dmtet_geometry.renderer
        self.camera_model = self.renderer.camera

    def render_image(self, w_geo, w_tex, camera):
        synthesis_kwargs = {"noise_mode": "const",
                            "update_geo":None,
                            "update_emas":None
                            }
        
        results = self.generator.generate(ws_tex=w_tex, ws_geo=w_geo, camera=camera, **synthesis_kwargs)
        
        result_dict = {"RGB": results[0],
                       "antilias_mask": results[1],
                       "mesh":{"vert": results[5],
                               "face":results[6]},
                       "mask_pyramid": results[9],
                       "tex_hard_mask": results[10],
                       "render_buffer": results[12],
                       "depth_buffer": torch.stack(results[12]["depth"]).squeeze(-1)}
        RGB = results[0]
        return RGB, result_dict
    
    def generate_w_tex(self, z_tex):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w_tex = self.mapping_tex(z=z_tex, c=torch.tensor(1).to(device), truncation_psi=0.7)
        return w_tex
    
    def generate_w_geo(self, z_geo):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w_geo = self.mapping_geo(z=z_geo, c=torch.tensor(1).to(device), truncation_psi=0.7)
        return w_geo
    
    def get_mesh(self, ws_geo, ws_tex):
        sdf_feature, tex_feature = self.generator.generator.get_feature(
            ws_tex[:, :self.generator.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :self.generator.generator.tri_plane_synthesis.num_ws_geo])
        ws_tex = ws_tex[:, self.generator.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, self.generator.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.generator.get_geometry_prediction(ws_geo, sdf_feature)
        return mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss
    
    def get_texture(self, ws_geo, ws_tex):
        texture = 0.
        return texture
    
    def forward(self, z_geo, z_tex, camera):
        w_geo = self.generate_w_geo(z_geo)
        w_tex = self.generate_w_tex(z_tex)
        RGB, result_dict = self.render_image(w_geo, w_tex, camera)
        return RGB, w_geo, w_tex, result_dict