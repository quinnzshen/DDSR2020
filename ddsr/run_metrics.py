from third_party.monodepth2.layers import compute_depth_errors
from third_party.monodepth2.utils import readlines
from torch.utils.data import DataLoader
from kitti_dataset import KittiDataset

def run_metrics(config_path, epoch):
    MIN_DEPTH = 0.1
    MAX_DEPTH = 100
    # Load data from config
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.Loader)
            
    weights_folder = os.path.join("models", self.log_path, "weights_{epoch}")
    print("-> Loading weights from {weights_folder}")
    encoder_path = os.path.join(weights_folder, "resnet_encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth_decoder.pth")

    encoder_dict = torch.load(encoder_path)

        dataset = KittiDataset.init_from_config(config["train_config_path"])
        dataloader = DataLoader(dataset, config["batch_size"], shuffle=False, collate_fn=self.collate, num_workers=config["num_workers"])

        encoder = networks.ResnetEncoder(config["num_layers"], False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["stereo_left_image"].to(self.device).float()

                output = depth_decoder(encoder(inputs))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    gt_path = os.path.join(config["gt_path"], "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_depth_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")