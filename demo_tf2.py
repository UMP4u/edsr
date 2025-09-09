import os, argparse, numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm 

# —— 显存按需增长 ——
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

def load_img(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, np.float32) / 255.0
    return arr * 2.0 - 1.0

def save_img(sr_arr, out_path):
    sr = (np.clip(sr_arr, -1, 1)*0.5 + 0.5)
    sr_uint8 = (sr*255).astype(np.uint8)
    Image.fromarray(sr_uint8).save(out_path)

def sr_folder(model, lr_dir, save_dir, ext):
    os.makedirs(save_dir, exist_ok=True)
    for fn in tqdm(sorted(os.listdir(lr_dir)),desc="processiong images"):
        if not fn.lower().endswith(ext.lower()): continue
        lr_path  = os.path.join(lr_dir, fn)
        out_path = os.path.join(save_dir, fn)
        # (1) 读图
        lr_np = load_img(lr_path)[None,...]    # [1, H, W, 3]
        # (2) 直接 Eager 推理
        sr_np = model(lr_np, training=False).numpy()[0]
        # (3) 保存
        save_img(sr_np, out_path)
        print("Saved", fn)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arc',        required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--lr_dir',     required=True)
    p.add_argument('--save_dir',   required=True)
    p.add_argument('--ext',        default='.png')
    p.add_argument('--cuda',       default='0')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # 载入模型
    from model import get_generator
    model = get_generator(args.arc, is_train=False)
    model.load_weights(args.model_path)
    # 批量超分
    sr_folder(model, args.lr_dir, args.save_dir, args.ext)

if __name__ == '__main__':
    main()
