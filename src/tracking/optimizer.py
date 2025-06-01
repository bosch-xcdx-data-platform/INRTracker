import torch
import torch.nn as nn
from torchvision.transforms import Resize
import numpy as np
import lpips

LPIPS_FN = lpips.LPIPS(net='vgg', spatial=False).eval()
MSE_FN = nn.MSELoss()

def get_lpips_loss(pred_image, gt_image, lpips_loss_fn, use_lpips_first_two=True, use_lpips_last_three=True):
    lpips_loss = 0.
    
    # clamp to be in 0, 1 range
    pred_image = torch.clamp(pred_image, 0., 1.)
    gt_image = torch.clamp(gt_image, 0., 1.) 
    
    
    if not np.array(list(pred_image.shape[1:])).all() > 16:
        if pred_image.shape[1] == 0:
            # print("Warning: object imploded!")
            return 0.
        else:
            min_size = np.min(list(pred_image.shape[1:]))
            scale_factor = int(np.ceil(16 / min_size))
            resize_to_16 = Resize(list(torch.tensor(pred_image.shape[1:]) * scale_factor))
            pred_image = resize_to_16(pred_image)
            gt_image = resize_to_16(gt_image)
    
    # Define which layers to use
    layer_ids = []
    if use_lpips_first_two: 
        # print("Using LPIPS first two layers\n")
        layer_ids.extend([0, 1])
    if use_lpips_last_three:
        # print("Using LPIPS last three layers\n")
        layer_ids.extend([2, 3, 4])
    
    # Compute LPIPS feature maps
    _, res = lpips_loss_fn(pred_image, gt_image, retPerLayer=True, normalize=True) 
    
    # Compute loss from relevant feature maps
    lpips_loss = torch.cat(res)[layer_ids].mean()

    return lpips_loss

def get_params_optimizer(tracklets, frame=0, optimize_w=False, cam_id=None, 
                         tex_only_optim_adam=True, 
                         optim_kwargs_tex_only={}):
    optimizer_all = []
    optimizer_all_tex = []
    
    all_params_name = ['sdf', 'map', 'tex', 'R', 'T', 'S']
    
    embedd_space = 'w' if optimize_w else 'z'
    # counter = 0
    # target_car_idx = 1
    for key, track in tracklets.items():
        
        # if counter != target_car_idx:
        #     continue
        # counter += 1
        
        if 'optimizer' in track:
            optimizer_k = track['optimizer']
        else:            
            # LBFGS TEX PARAMS
            codes_tex_lbfgs  = track[frame][f'{embedd_space}_tex']
            codes_sdf_lbfgs  = track[frame][f'{embedd_space}_shape']
            codes_map_lbfgs  = track[frame]['z_map']
            rotation_lbfgs   = track[frame]['heading']
            shift_lbfgs      = track[frame]['translation']
            scale_lbfgs      = track[frame]['scale']
            
            # OTHER
            codes_sdf = nn.ParameterDict({'sdf_{}_{}'.format(key, frame): track[frame][f'{embedd_space}_shape']})
            codes_map = nn.ParameterDict({'map_{}_{}'.format(key, frame): track[frame]['z_map']})
            codes_tex = nn.ParameterDict({'tex_{}_{}'.format(key, frame): codes_tex_lbfgs })
            rotation = nn.ParameterDict({'heading_{}_{}'.format(key, frame): track[frame]['heading']})
            shift = nn.ParameterDict({'translation_{}_{}'.format(key, frame): track[frame]['translation']})
            scale = nn.ParameterDict({'scale_{}_{}'.format(key, frame): track[frame]['scale']})
            
            
            optim_params_3Dmodel = [{"params": codes_sdf.values(), "lr": 0},
                                    {"params": codes_map.values(), "lr": 0},
                                    {"params": codes_tex.values(), "lr": 0}]

            optim_params_pose = [{"params": rotation.values(), "lr": 0},
                                 {"params": shift.values(), "lr": 0},
                                 {"params": scale.values(), "lr": 0}]
            
            optim_params_tex_only = [codes_tex_lbfgs, shift_lbfgs, scale_lbfgs]
            all_params = optim_params_3Dmodel + optim_params_pose
            optimizer_k = torch.optim.Adam(all_params)
            
            if tex_only_optim_adam:
                optimizer_k_tex = torch.optim.Adam(optim_params_tex_only, **optim_kwargs_tex_only)
            else:
                optimizer_k_tex = torch.optim.LBFGS(optim_params_tex_only, **optim_kwargs_tex_only)
            
        # TEXONLY PARAMS SET
        optimizer_all_tex.append(optimizer_k_tex)
        # OTHERS
        optimizer_all.append(optimizer_k)
        
    return optimizer_all, optimizer_all_tex, all_params_name


def compute_loss(rgb_pred, rgb_gt, fg_mask, 
                 gen_patches, gt_patches,
                 w_geo, w_tex, w_avg_geo, w_avg_tex,
                 truncation_psi=0.7, truncation_alpha_tex=1.0,  truncation_alpha_geo=1.0, truncation_cutoff=None,
                 use_mse=True, use_lpips_first_two=False, use_lpips_last_three=True, lpips_kwargs={}, 
                 **kwargs):
    
    device = rgb_pred.device    
    loss = torch.tensor([0.], device=rgb_pred.device)
    
    # RGB LOSS
    rgb_loss = torch.tensor([0.], device=device)
    if fg_mask.sum() > 0. and use_mse == True: 
        rgb_loss = MSE_FN(rgb_pred[fg_mask], rgb_gt[fg_mask])
    else:
        rgb_loss = torch.tensor([0.], device=device)
    loss += 2.0*rgb_loss
    
    # LEARNED PERCEPTUAL LOSS
    lpips_loss = torch.tensor([0.], device=device)
    if use_lpips_last_three or use_lpips_first_two:
        # TODO: Explore getting rid of the for loop
        for i in range(len(gt_patches)):
            lpips_loss += get_lpips_loss(gen_patches[i].permute(2, 0, 1), gt_image=gt_patches[i].permute(2, 0, 1),
                                    lpips_loss_fn=LPIPS_FN.to(device), use_lpips_first_two=use_lpips_first_two, 
                                    use_lpips_last_three=use_lpips_last_three) * lpips_kwargs["alpha"]
        # Mean over number of patches/objects
        lpips_loss = lpips_loss / len(gt_patches) 
    loss += lpips_loss
    
    # RESGULARIZE EMBEDDINGS to be inside embedding distribution
    # 1. Compute truncated result
    truncation_loss_geo = torch.tensor([0.], device=device)
    truncation_loss_tex = torch.tensor([0.], device=device)
    
    w_trunc_geo = apply_truncation_trick(w_geo, w_avg_geo, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    w_trunc_tex = apply_truncation_trick(w_tex, w_avg_tex, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    # 2. Compute loss as difference from truncated result
    truncation_loss_geo = torch.mean(torch.abs(w_geo - w_trunc_geo)) * truncation_alpha_geo if w_trunc_geo is not None else torch.tensor([0.], device=device)
    truncation_loss_tex = torch.mean(torch.abs(w_tex - w_trunc_tex)) * truncation_alpha_tex if w_trunc_tex is not None else torch.tensor([0.], device=device)
    truncation_loss = truncation_loss_geo + truncation_loss_tex
    loss += truncation_loss
    
    return loss, rgb_loss, lpips_loss, truncation_loss, truncation_loss_geo, truncation_loss_tex

def get_patch(gen_img, gt_img, instance_mask, padding_ratio=0.5):
    device=gen_img.device
    H, W = instance_mask.shape
    padding = int(W * padding_ratio/100)
    n_patches = instance_mask.max()+1
    
    vu_grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))).to(device)
    
    n_instance_mask = instance_mask[..., None].repeat(1,1,n_patches)
    mask_filter = torch.arange(0, n_patches).expand(n_instance_mask.shape).to(device)
    patch_bool = (n_instance_mask == mask_filter)
    patch_bool_vu = patch_bool[None].repeat(2,1,1,1)
    
    index_mesh = patch_bool_vu * vu_grid[..., None]
    
    max_v = torch.minimum(
        index_mesh[0].view(-1, n_patches).max(0)[0] + padding, 
        torch.tensor(H, dtype=torch.int, device=device))
    max_u = torch.minimum(
        index_mesh[1].view(-1, n_patches).max(0)[0] + padding, 
        torch.tensor(W, dtype=torch.int, device=device))

    index_mesh[~patch_bool_vu] = np.maximum(H,W)
    min_v = torch.maximum(
        index_mesh[0].view(-1, n_patches).min(0)[0] - padding,
        torch.zeros(1, dtype=torch.int, device=device))
    min_u = torch.maximum(
        index_mesh[1].view(-1, n_patches).min(0)[0] - padding,
        torch.zeros(1, dtype=torch.int, device=device))
    
    gt_patches = []
    gen_patches = []
    for i in range(n_patches):
        gt_patch = gt_img[min_v[i]:max_v[i], min_u[i]:max_u[i], ...]
        gen_patch = gen_img[min_v[i]:max_v[i], min_u[i]:max_u[i], ...]
        gt_patches.append(gt_patch)
        gen_patches.append(gen_patch)
        
    return gen_patches, gt_patches

def get_optim_state(k, optimizer, opti_config, w_avg_geo, w_avg_tex):
    steps_per_iter = opti_config['steps_in_iter']
    if k >= steps_per_iter:
        k = k - (steps_per_iter * (k // steps_per_iter))

    for optim_k in optimizer:
        if not isinstance(optim_k, list):
            optim_k, no_volume, use_mse, use_lpips_first_two, use_lpips_last_three = set_for_param_group(k, opti_config, optim_k, w_avg_geo, w_avg_tex)
        else:
            # for optim in optim_k:
            optim_k, no_volume, use_mse, use_lpips_first_two, use_lpips_last_three = set_for_single(k, opti_config, optim_k)
            
    return use_mse, use_lpips_first_two, use_lpips_last_three, 

def apply_truncation_trick(x, w_avg, num_ws=None, truncation_psi=0.7, truncation_cutoff=None):
    # Apply truncation (adpated from GET3D/training/geometry_predictor.py)
    if truncation_psi != 1:
        assert truncation_cutoff is None
        if num_ws is None or truncation_cutoff is None:
            x = w_avg.lerp(x, truncation_psi)
        else:
            x[:, :truncation_cutoff] = w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
    return x

def set_for_param_group(k, opti_config, optim, w_avg_geo, w_avg_tex):
    no_volume = True
    
    use_mse = True
    use_lpips_first_two = True
    use_lpips_last_three = True
    
    if opti_config['sdf']['start'] <= k < opti_config['sdf']['end']:
        optim.param_groups[0]['lr'] = opti_config['sdf']['lr']
        no_volume = False
        use_mse = False
        use_lpips_first_two = False
        # apply truncation trick
        # optim.param_groups[0]['params'][0].data = apply_truncation_trick(x=optim.param_groups[0]['params'][0].data, 
        #                                                                  w_avg=w_avg_geo, 
        #                                                                  num_ws=num_ws_geo,
        #                                                                  truncation_psi=opti_config['truncation_psi'])
    else:
        optim.param_groups[0]['lr'] = 0.

    if 'map' in opti_config:
        if opti_config['map']['start'] <= k < opti_config['map']['end']:
            optim.param_groups[1]['lr'] = opti_config['map']['lr']
        else:
            optim.param_groups[1]['lr'] = 0.

    if opti_config['tex']['start'] <= k < opti_config['tex']['end']:
        optim.param_groups[2]['lr'] = opti_config['tex']['lr']
         # apply truncation trick
        # optim.param_groups[2]['params'][0].data = apply_truncation_trick(x=optim.param_groups[2]['params'][0].data, 
        #                                                                      w_avg=w_avg_tex, 
        #                                                                      num_ws=num_ws_tex,
        #                                                               truncation_psi=opti_config['truncation_psi'])
    else:
        optim.param_groups[2]['lr'] = 0.
        
    # if opti_config['tex']['end'] == k:
    #     optim.param_groups[2]['params'][0].data = apply_truncation_trick(x=optim.param_groups[2]['params'][0].data, 
    #                                                                          w_avg=w_avg_tex, 
    #                                                                          num_ws=num_ws_tex,
    #                                                                   truncation_psi=opti_config['truncation_psi'])

    if opti_config['R']['start'] <= k < opti_config['R']['end']:
        optim.param_groups[3]['lr'] = opti_config['R']['lr']
        use_mse = False
        use_lpips_first_two = False
    else:
        optim.param_groups[3]['lr'] = 0.

    if opti_config['T']['start'] <= k < opti_config['T']['end']:
        optim.param_groups[4]['lr'] = opti_config['T']['lr']
        use_mse = False
        use_lpips_first_two = False
    else:
        optim.param_groups[4]['lr'] = 0.

    if opti_config['S']['start'] <= k < opti_config['S']['end']:
        optim.param_groups[5]['lr'] = opti_config['S']['lr']
        use_mse = False
        use_lpips_first_two = False
        no_volume = False
    else:
        optim.param_groups[5]['lr'] = 0
    return optim, no_volume, use_mse, use_lpips_first_two, use_lpips_last_three


def set_for_single(k, opti_config, optim, w_avg_geo, w_avg_tex):
    no_volume = True
    
    use_mse = True
    use_lpips_first_two = False
    use_lpips_last_three = True

    if opti_config['sdf']['start'] <= k < opti_config['sdf']['end']:
        optim[0].param_groups[0]['lr'] = opti_config['sdf']['lr']
        no_volume = False
        
        use_mse = False
        # apply truncation trick
        # optim.param_groups[0]['params'][0].data = apply_truncation_trick(x=optim.param_groups[0]['params'][0].data, 
        #                                                                  w_avg = w_avg_geo, 
        #                                                                  num_ws = num_ws_geo,
        #                                                                  truncation_psi=opti_config['truncation_psi'])
    else:
        optim[0].param_groups[0]['lr'] = 0.

    if 'map' in opti_config:
        if opti_config['map']['start'] <= k < opti_config['map']['end']:
            optim[1].param_groups[0]['lr'] = opti_config['map']['lr']
        else:
            optim[1].param_groups[0]['lr'] = 0.

    if opti_config['tex']['start'] <= k < opti_config['tex']['end']:
        optim[2].param_groups[0]['lr'] = opti_config['tex']['lr']
        
        # apply truncation trick
        # optim.param_groups[2]['params'][0].data = apply_truncation_trick(x=optim.param_groups[2]['params'][0].data, 
        #                                                                      w_avg=w_avg_tex, 
        #                                                                      num_ws=num_ws_tex,
        #                                                                      truncation_psi=opti_config['truncation_psi'])
        use_lpips_first_two = True
    else:
        optim[2].param_groups[0]['lr'] = 0.

    if opti_config['R']['start'] <= k < opti_config['R']['end']:
        optim[3].param_groups[0]['lr'] = opti_config['R']['lr']
    else:
        optim[3].param_groups[0]['lr'] = 0.

    if opti_config['T']['start'] <= k < opti_config['T']['end']:
        optim[4].param_groups[0]['lr'] = opti_config['T']['lr']
    else:
        optim[4].param_groups[0]['lr'] = 0.

    if opti_config['S']['start'] <= k < opti_config['S']['end']:
        optim[5].param_groups[0]['lr'] = opti_config['S']['lr']
        no_volume = False
    else:
        optim[5].param_groups[0]['lr'] = 0

    return optim, no_volume, use_mse, use_lpips_first_two, use_lpips_last_three


def move_tracklets_to_device(tracklets, current_fr_id, embedd_space='w', device='cuda:0'):
    if 'cuda' not in device:
        is_cuda = False
    else:
        is_cuda = True
    
    for obj_key, obj_i in tracklets.items():
        if current_fr_id in obj_i:
            if not obj_i[current_fr_id]['heading'].is_cuda == is_cuda:
                obj_i[current_fr_id]['heading'] = obj_i[current_fr_id]['heading'].to(device)
            if not obj_i[current_fr_id]['translation'].is_cuda == is_cuda:
                obj_i[current_fr_id]['translation'] = obj_i[current_fr_id]['translation'].to(device)
            if not obj_i[current_fr_id]['scale'].is_cuda == is_cuda:
                obj_i[current_fr_id]['scale'] = obj_i[current_fr_id]['scale'].to(device)
            if not obj_i[current_fr_id][f'{embedd_space}_shape'].is_cuda == is_cuda:
                obj_i[current_fr_id][f'{embedd_space}_shape'] = obj_i[current_fr_id][f'{embedd_space}_shape'].to(device)
            if not obj_i[current_fr_id][f'{embedd_space}_tex'].is_cuda == is_cuda:
                obj_i[current_fr_id][f'{embedd_space}_tex'] = obj_i[current_fr_id][f'{embedd_space}_shape'].to(device)
    
    return tracklets