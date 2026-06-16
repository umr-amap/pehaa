import glob
import json
import os
import numpy as np
import warnings
from typing import Any, Callable, Dict ,Optional

warnings.filterwarnings("ignore", message="Can't initialize NVML")
import logging

import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import MLP
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast

from torchgeo.datasets import stack_samples
from torchgeo.datasets import RasterDataset
from torchgeo.datasets import Sentinel2, BoundingBox
from torchgeo.samplers import RandomGeoSampler
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K
from torchvision.transforms import v2
import geopandas as gpd

import timm
from timm import create_model
import peft
import hydra
from omegaconf import DictConfig


from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


class ROIDataset(RasterDataset):

    def __init__(
            self, 
            raster_dataset: RasterDataset,
            gdf: gpd.GeoDataFrame, 
            target_var, 
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            bboxes_col:str = 'bboxes',
            ):

        self.dataset = raster_dataset
        self.target_var = target_var
        self.gdf = gdf
        self.transforms = transforms
        self.bboxes_col = bboxes_col

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        """Retrieve image/mask and metadata indexed by query.

        Args:
            index: Index of sample to fetch

        Returns:
            sample of image/mask, metadata and ground truth at that index

        Raises:
            IndexError: if query is not found in the index
        """
        query = self.gdf.iloc[idx][self.bboxes_col]

        sample = self.dataset[query]

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        if isinstance(self.target_var, list):
            for gt in self.target_var:
                sample[gt] = self.gdf.iloc[idx][gt]
        else :
            sample[self.target_var] = self.gdf.iloc[idx][self.target_var]

        return sample


### disable time information to overlap
### cf. https://github.com/torchgeo/torchgeo/issues/2571
class Sentinel2SpatialOnly(Sentinel2):
    filename_regex = r"""                                                                
        ^T(?P<tile>\d{{2}}[A-Z]{{3}})                                                    
        _(\d{{8}}T\d{{6}})                                                       
        _(?P<band>B[018][\dA])                                                           
        (?:_(?P<resolution>{}m))?                                                        
        \..*$                                                                            
    """ 
    all_bands = [
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B11',
            'B12',
            ]


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17,device="cuda"):
        super().__init__()

        self.device = device
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=self.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class DOFAEncoder(nn.Module):
    def __init__(self, backbone, emb_dim=768, proj_layers = [512, 2048, 2048, 128]):
        super().__init__()
        self.backbone = backbone
        self.proj = MLP(emb_dim, proj_layers, norm_layer=nn.BatchNorm1d)
        self.wavelengths = [
            0.560,
            0.665,
            0.705,
            0.740,
            0.783,
            0.842,
            1.375,
            1.610,
            2.190,
            ]

    def forward(self, x):
        """
        detail of the computations and shapes
        #### (samples, versions, C, W, H)
        N, V = x.shape[:2]
        #### (samples x versions, C, W, H)
        x_flat = x.flatten(0, 1)
        #### (samples x versions, embeddings)
        emb = self.backbone(x_flat)
        #### (versions, samples, projection)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        ### (versions, samples, embeddings)
        emb = emb.reshape(N, V, -1).transpose(0, 1)
        return emb, proj

        DOFA takes additionnal parameter wavelengths in the forward

        """
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1), self.wavelengths)
        return emb.reshape(N, V, -1).transpose(0, 1), self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class ViTEncoder(DOFAEncoder):
    def __init__(self, backbone, emb_dim=768, proj_layers=[512, 2048, 2048, 128]):
        super().__init__(backbone, emb_dim, proj_layers)

    def forward(self, x):
        N, V = x.shape[:2]
        x_flat = x.flatten(0, 1)
        emb = self.backbone(x_flat)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        emb = emb.reshape(N, V, -1).transpose(0, 1)
        return emb, proj


def prepare_model(arch, in_chans, device, do_peft=True, r=8, nproj_layers=3, pretrained=True):

    ### if model is not pretrained, no need to do PEFT
    if pretrained == False:
        do_peft = False

    if arch == 'dofa':
        encoder = dofa_base_patch16_224(
                weights=DOFABase16_Weights.DOFA_MAE, 
                num_classes=0
                )
        h,w = 224, 224

    else:
        encoder = create_model(
                arch, 
                pretrained=pretrained, 
                in_chans=in_chans, 
                num_classes=0
                )
        ## get expected input for each model via timm 
        data_config = timm.data.resolve_model_data_config(encoder)
        (_, h, w) = data_config["input_size"]


    if do_peft:

        accepted_types = [
          torch.nn.Linear, 
          torch.nn.Embedding, 
          torch.nn.Conv1d, 
          torch.nn.Conv2d, 
          torch.nn.Conv3d, 
          torch.nn.MultiheadAttention
          ]

        target_modules = []
        for n, m in encoder.named_modules():
            if type(m) in accepted_types:
                if not (('fc' in n) or ('head' in n)):
                    target_modules.append(n)

        config = peft.LoraConfig(r=r, target_modules=target_modules)
        peft_encoder = peft.get_peft_model(encoder, config).to(device)


    else:
        peft_encoder = encoder

    proj_layers = [512] + (nproj_layers - 1)*[2048] + [128]
    print(proj_layers)
    
    ## get embed_dim with single inference
    if arch == 'dofa':
        model = DOFAEncoder(peft_encoder, proj_layers=proj_layers).to(device)

    else:
        model = ViTEncoder(peft_encoder, proj_layers=proj_layers).to(device)

    return model, h, w


def plot_rgb_grid(tensor, rgb_bands=[1, 0, 2], out_path='./samples.png'):
    """
    Plot RGB images in a grid.

    Args:
        tensor (torch.Tensor): Tensor of shape (batch, versions, channels, height, width)
        rgb_bands (list): Indices of RGB bands in the channel dimension.
    """
    batch_size, num_versions, _, height, width = tensor.shape
    if batch_size > 6: ## avoid plotting 256 rows
        batch_size = 6
    fig, axes = plt.subplots(batch_size, num_versions, figsize=(num_versions * 3, batch_size * 3))

    if batch_size == 1 and num_versions == 1:
        axes = axes.reshape(1, 1)

    for b in range(batch_size):
        for v in range(num_versions):
            # Select RGB channels and permute to (height, width, channels)
            rgb = tensor[b, v, rgb_bands, :, :].permute(1, 2, 0)
            # Normalize to [0, 1] for plotting
            rgb = torch.where(rgb > 1.5, 1.5, rgb)
            rgb = rgb.cpu().numpy()
            # rgb = torch.where(rgb < 0.7, 0.7, rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

            ax = axes[b, v] if batch_size > 1 and num_versions > 1 else axes
            ax.imshow(rgb)
            ax.axis('off')
            # ax.annotate
            if b == 0:
                ax.set_title(f"Raster {v}", size=20)
            if v == 0:
                ax.annotate(f"Sample {b}", (0, 0.5), xytext=(-20, 0),
                            textcoords='offset points', xycoords='axes fraction',
                            ha='right', va='center', rotation=90,
                            size=20)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def find_s2_img_dirs(root_path):
    """
    Find all IMG_DATA directories inside .SAFE folders.

    Args:
        root_path (str): Path to the directory containing .SAFE folders.

    Returns:
        list: List of paths to all IMG_DATA directories.
    """
    img_data_dirs = []
    # Find all .SAFE directories
    safe_dirs = glob.glob(os.path.join(root_path, "*.SAFE"))

    for safe_dir in safe_dirs:
        # Find IMG_DATA directory inside each .SAFE/GRANULE/*/IMG_DATA
        granule_dirs = glob.glob(os.path.join(safe_dir, "GRANULE", "*"))
        for granule_dir in granule_dirs:
            img_data_path = os.path.join(granule_dir, "IMG_DATA")
            if os.path.exists(img_data_path):
                img_data_dirs.append(img_data_path)

    return img_data_dirs


def evaluate(dataloader,model, device, in_chans, logger):

    model.eval()

    embs = []
    for i, vs in enumerate(dataloader):
        logger.info(f'val {i}/{len(dataloader)}')

        with autocast('cuda', dtype=torch.bfloat16):
            vs = vs['image'].to(device, non_blocking=True)
            vs = vs.reshape(vs.shape[0],int(vs.shape[1]/in_chans),in_chans, vs.shape[2], vs.shape[3])
            # plot_rgb_grid(vs, out_path='./temp_val.png')

            emb, proj = model(vs)
            ### (versions, samples, dimensions)
            embs.append(emb.detach().cpu())
            del vs
            del emb
    
    ####(versions, batch*samples, dimensions)
    x = torch.concat(embs, dim=1 )
    ### (versions x batch, dimensions)
    x = x.flatten(start_dim=0, end_dim=-2)
    ### raster 1 at the begining, raster 2 at the end

    model.train()
    return x


def evaluate_supervised(dataloader,model, device, in_chans, logger, gt_key='Id'):

    model.eval()

    embs0 = []
    embs1 = []
    gts = []
    for i, vs in enumerate(dataloader):
        logger.info(f'val inf {i}/{len(dataloader)}')

        with autocast('cuda', dtype=torch.bfloat16):

            gts.extend(vs[gt_key])
            vs = vs['image'].to(device, non_blocking=True)
            vs = vs.reshape(vs.shape[0],int(vs.shape[1]/in_chans),in_chans, vs.shape[2], vs.shape[3])
            plot_rgb_grid(vs, out_path='./temp_val.png')

            emb, proj = model(vs)
            embs0.extend(emb[0,::].detach().cpu())
            embs1.extend(emb[1,::].detach().cpu())

    embs0 = np.asarray(embs0)
    embs1 = np.asarray(embs1)
    gts = np.asarray(gts)
    skf = StratifiedKFold(n_splits=5)
    tp = 0

    ### train knn on one raster but predict on the other
    for i, (train_index, test_index) in enumerate(skf.split(embs0, gts)):
        
        knn = KNeighborsClassifier()
        knn.fit(embs0[train_index], gts[train_index])
        preds = knn.predict(embs1[test_index])
        gt = gts[test_index]
        tp += np.sum(preds == gt)

    acc = tp / gts.shape[0]
    
    model.train()
    return acc, (embs0, embs1)


def prepare_supervised_dataloader(
        raster_paths, 
        shapefile_path, 
        sampling_size_px, 
        means, 
        stds, 
        batch_size, 
        target_var,
        raster_indexes=[-1,-2],
        num_workers=10,
        ):

    val_raster1 = raster_paths[raster_indexes[0]]
    val_raster2 = raster_paths[raster_indexes[1]]
    val_transforms = AugmentationSequential(
            K.Resize((224,224)),
            K.Normalize(means,stds),
            data_keys=None,
            keepdim=True,
            )
    val_dataset = Sentinel2SpatialOnly(val_raster1, transforms=val_transforms) & Sentinel2SpatialOnly(val_raster2, transforms=val_transforms)


    gdf = gpd.read_file(shapefile_path)
    gdf.to_crs(val_dataset.crs)
    gdf.geometry = gdf.geometry.centroid
    gdf['buff'] = gdf.buffer(sampling_size_px * val_dataset.res[0], cap_style=3)
    gdf['bboxes'] = gdf['buff'].apply(
            lambda geom: BoundingBox(
                geom.bounds[0], 
                geom.bounds[2], 
                geom.bounds[1], 
                geom.bounds[3], 
                val_dataset.bounds[-2], 
                val_dataset.bounds[-1]
                )
            )
    gdf.to_crs(val_dataset.crs)
    roidataset = ROIDataset(val_dataset, gdf, target_var=target_var)
    dataloader = DataLoader(roidataset, batch_size=batch_size, collate_fn=stack_samples, num_workers=num_workers)

    return dataloader


@hydra.main(version_base=None, config_path='./conf/', config_name='config')
def main(cfg: DictConfig):

    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
    hydra_outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    outdir = f"{hydra_outdir}/{task_id}"
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger()
    logging.basicConfig(filename=os.path.join(outdir,'main.log'), encoding='utf-8', level=logging.DEBUG)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator()
    rng.manual_seed(cfg.param.seed)

    paths = sorted(find_s2_img_dirs(cfg.data.root_data_dir))
    # del paths[-2] ### missing bands in second to last raster, not used.
    logger.info(cfg.param.arch)
    logger.info(outdir)
    logger.info(paths)
    root_val_paths = sorted(find_s2_img_dirs(cfg.data.sup_data_dir))
    shapefile_path = [os.path.join(cfg.data.sup_data_dir, f) for f in os.listdir(cfg.data.sup_data_dir) if f.endswith('.shp')]
    val_raster1 = root_val_paths[-1]
    val_raster2 = root_val_paths[-2]
    logger.info(f'val paths : {val_raster1}{val_raster2}')

    BAND_STATS = {
    'mean': {
        'B03': 1041.8842963,
        'B04': 946.554,
        'B05': 1199.18896296,
        'B06': 2003.00696296,
        'B07': 2374.00874074,
        'B08': 2301.22014815,
        'B8A': 2599.78311111,
        'B11': 1820.69659259,
        'B12': 1118.20259259
    },
    'std': {
        'B03': 684.77615743,
        'B04': 620.02902871,
        'B05': 791.86263829,
        'B06': 1341.28018273,
        'B07': 1595.39989386,
        'B08': 1545.52915718,
        'B8A': 1750.12066835,
        'B11': 1216.48651476,
        'B12': 736.6981037
    }
    }


    means = list(BAND_STATS['mean'].values())
    stds = list(BAND_STATS['std'].values())

    model, h, w = prepare_model(
            arch=cfg.param.arch, 
            nproj_layers=cfg.param.nproj_layers, 
            device=device,
            in_chans=len(means),
            do_peft=cfg.param.peft,
            r=cfg.param.rank,
            pretrained=cfg.param.pretrained,
            )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of trainable parameters: {total_params}')


    transforms = AugmentationSequential(
            v2.RandomResize(min_size=w, max_size=w*5),
            v2.Normalize(means,stds), # normalize occurs only on raster, not mask
            v2.CenterCrop((h,w)),
            data_keys=None,
            keepdim=True,
            )

    val_dataloader = prepare_supervised_dataloader(
            raster_paths=root_val_paths, 
            shapefile_path=shapefile_path[0], 
            sampling_size_px=cfg.param.sampling_size, 
            means=means, 
            stds=stds, 
            batch_size=cfg.param.batch_size, 
            target_var='Id',
            )

    dataset = None

    ## stack all RasterDatasets on top of one an other
    for path in paths:
        if dataset:
            raster = Sentinel2SpatialOnly(paths = [path])
            dataset = dataset & raster
            raster.transforms = transforms
        else:
            dataset = Sentinel2SpatialOnly(paths = [path])
            dataset.transforms = transforms

    sampler = RandomGeoSampler(
            dataset, 
            size=cfg.param.sampling_size, 
            length=cfg.param.samples, 
            generator=rng
            )
    dataloader = DataLoader(
            dataset, 
            batch_size=cfg.param.batch_size, 
            num_workers=cfg.param.num_workers, 
            sampler=sampler,
            collate_fn=stack_samples
            )

    opt = Adam(model.parameters(), lr=cfg.param.lr, weight_decay=1e-6)

    sigreg = SIGReg(device=device).to(device)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100)

    scaler = GradScaler(enabled="cuda" == "cuda")
    
    metrics_logs = {}
    losses = []
    lrs = []
    l2s = []
    inter_sim_means = []
    inter_sim_stds = []
    intra_sim_means = []
    intra_sim_stds = []
    accs = []

    model.train()
    for i, vs in enumerate(dataloader):

        with autocast("cuda", dtype=torch.bfloat16):
            #### (Batch, versions x C, W, H)
            vs = vs['image'].to(device, non_blocking=True)

            #### (Batch, versions, C, W, H)
            #### should be ok as it's correctly plotted afterwards
            vs = vs.reshape(vs.shape[0],int(vs.shape[1]/len(means)),len(means), vs.shape[2], vs.shape[3])

            if (i%cfg.param.eval_freq == 0) or (i in [10,20,30,40]) :
                
                # x = evaluate(
                #         dataloader=val_dataloader, 
                #         model=model,
                #         device=device,
                #         in_chans=len(means),
                #         logger=logger,
                #         )
                acc, (x1,x2)= evaluate_supervised(
                        dataloader=val_dataloader, 
                        model=model,
                        device=device, 
                        in_chans=len(means),
                        logger=logger,
                        )

                np.save(file = os.path.join(outdir, f'emb-v1-{i:06}.npy'), arr=x1)
                np.save(file = os.path.join(outdir, f'emb-v2-{i:06}.npy'), arr=x2)
                ### cosine and l2
                # x1, x2 = x.chunk(2)
                ### temporary and inefficent
                x1, x2 = torch.Tensor(x1), torch.tensor(x2)
                intercos = nn.CosineSimilarity(dim=1, eps=1e-6)
                inter_sim = intercos(x1,x2)
                inter_sim_means.append(inter_sim.mean().item())
                inter_sim_stds.append(inter_sim.std().item())
                l2 = torch.cdist(x1,x2).mean().item()
                accs.append(acc)

                ## saving for futur plots and viz
                # x_np = x.detach().float().cpu().numpy()
                # np.save(file = os.path.join(outdir, f'emb{i:06}.npy'), arr=x_np)

                metrics_logs['inter_sim_means'] = inter_sim_means
                metrics_logs['intra_sim_means'] = intra_sim_means
                metrics_logs['inter_sim_stds'] = inter_sim_stds
                metrics_logs['intra_sim_stds'] = intra_sim_stds
                metrics_logs['l2s'] = l2s
                metrics_logs['accs'] = accs

            if i%cfg.param.chkpt_freq == 0:

                logger.info('saving sample images')
                plot_rgb_grid(vs, out_path=os.path.join(outdir, f'samples{i:06}.png'))

                logger.info('saving model')
                if cfg.param.peft==True:
                    model.backbone.save_pretrained(os.path.join(outdir, f'./checkpoint{i:06}/'))
                else:
                    torch.save(model.state_dict(), os.path.join(outdir, f'./checkpoint{i:06}.pth'))

                logger.info('saving losses')
                metrics_logs['losses'] = losses
                metrics_logs['lrs'] = lrs

                with open(os.path.join(outdir, 'losses.json'), 'w') as f:
                    json.dump(metrics_logs, f)

            emb, proj = model(vs)
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = sigreg_loss * cfg.param.lamb + inv_loss * (1 - cfg.param.lamb)

            opt.zero_grad()
            scaler.scale(lejepa_loss).backward()
            scaler.step(opt)
            scaler.update()
            # scheduler.step()
            scheduler.step(lejepa_loss)

            losses.append(lejepa_loss.item())
            lrs.append(scheduler.get_last_lr()[0])

            logger.info(f'it: {i}/{len(dataloader)} loss: {lejepa_loss.item():4f} acc: {acc:2f} lr: {scheduler.get_last_lr()[0]:2f}')


if __name__ == "__main__":


    main()

    ### SSL4EO stats
    # BAND_STATS = {
    # 'mean': {
    #     'B01': 1353.72696296,
    #     'B02': 1117.20222222,
    #     'B03': 1041.8842963,
    #     'B04': 946.554,
    #     'B05': 1199.18896296,
    #     'B06': 2003.00696296,
    #     'B07': 2374.00874074,
    #     'B08': 2301.22014815,
    #     'B8A': 2599.78311111,
    #     'B09': 732.18207407,
    #     'B10': 12.09952894,
    #     'B11': 1820.69659259,
    #     'B12': 1118.20259259
    # },
    # 'std': {
    #     'B01': 897.27143653,
    #     'B02': 736.01759721,
    #     'B03': 684.77615743,
    #     'B04': 620.02902871,
    #     'B05': 791.86263829,
    #     'B06': 1341.28018273,
    #     'B07': 1595.39989386,
    #     'B08': 1545.52915718,
    #     'B8A': 1750.12066835,
    #     'B09': 475.11595216,
    #     'B10': 98.26600935,
    #     'B11': 1216.48651476,
    #     'B12': 736.6981037
    # }
    # }
