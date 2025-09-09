import os
import time
import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

import data
from model import get_generator
from metrics import psnr
from utils import save_params, num_iter_per_epoch
from callbacks import make_tb_callback, make_lr_callback, make_cp_callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_model(model, path):
    if path is not None:
        print("** Load model at: " + path)
        model.load_weights(path)
    return model


def make_exp_folder(exp_dir, model_name):
    folder = os.path.join(
        exp_dir,
        f"{model_name}-{datetime.datetime.now().strftime('%m-%d-%H:%M')}"
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def adaptive_batch_size(n_gpus):
    return 16 if n_gpus < 3 else 32


def prepare_model(**params):
    strategy = tf.distribute.MirroredStrategy()
    print("Number of GPUs: ", strategy.num_replicas_in_sync)
    with strategy.scope():
        model = get_generator(params['arc'])
        loss = (mean_squared_error
                if params['arc'] in ('srfeat', 'srgan')
                else mean_absolute_error)
        model = load_model(model, params['resume'])
        optimizer = Adam(learning_rate=params['lr_init'])
        model.compile(optimizer=optimizer, loss=loss, metrics=[psnr])
    return model


def train(**params):
    # 1. 加载训练和验证数据集
    print("** Loading training images")
    start = time.time()

    # —— 单行调用，防止语法错误 —— 
    lr_hr_ds, n_data = data.load_train_dataset(
        params['lr_dir'], params['hr_dir'], params['ext'],
        params['batch_size'], params['patch_size_lr'], params['patch_size_hr']
    )
    val_ds, n_val_data = data.load_test_dataset(
        params['val_lr_dir'], params['val_hr_dir'], params['val_ext'],
        params['val_batch_size'], params['patch_size_lr'], params['patch_size_hr']
    )

    print(f"Finish loading images in {time.time() - start:.2f}s")

    # 2. 构建并编译模型
    model = prepare_model(**params)

    # 3. 创建实验目录 & 回调
    exp_folder = make_exp_folder(params['exp_dir'], params['arc'])
    save_params(exp_folder, **params)
    tb_callback = make_tb_callback(exp_folder)
    lr_callback = make_lr_callback(
        params['lr_init'], params['lr_decay'], params['lr_decay_at_steps']
    )
    cp_callback = make_cp_callback(exp_folder, model)

    # 4. 开始训练
    model.fit(
        lr_hr_ds,
        epochs=params['epochs'],
        steps_per_epoch=num_iter_per_epoch(n_data, params['batch_size']),
        callbacks=[tb_callback, cp_callback, lr_callback],
        initial_epoch=params['init_epoch'],
        validation_data=val_ds,
        validation_steps=n_val_data
    )

    # 5. 保存最终模型权重并清理 session
    model.save_weights(os.path.join(exp_folder, 'final_model.h5'))
    K.clear_session()


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-Resolution')
    parser.add_argument('--arc',       type=str, required=True, help='Model type?')
    parser.add_argument('--train',     type=str, required=True, help='Path to training data')
    parser.add_argument('--train-ext', type=str, required=True, help='Extension of training images')
    parser.add_argument('--valid',     type=str, required=True, help='Path to validation data')
    parser.add_argument('--valid-ext', type=str, required=True, help='Extension of validation images')
    parser.add_argument('--resume',    type=str, default=None, help='Path to a checkpoint')
    parser.add_argument('--init_epoch',type=int, default=0,    help="Initial epoch")
    parser.add_argument('--cuda',      type=str, default=None, help='Comma-separated GPU ids')
    args = parser.parse_args()

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        n_gpus = len(args.cuda.split(','))
        batch_size = adaptive_batch_size(n_gpus)
    else:
        print('Training without gpu. It is recommended to use at least one gpu.')
        n_gpus = 0
        batch_size = 12

    params = {
        'arc':               args.arc,
        'resume':            args.resume,
        'init_epoch':        args.init_epoch,
        'n_gpus':            n_gpus,
        'epochs':            20,
        'lr_init':           1e-4,
        'lr_decay':          0.5,
        'lr_decay_at_steps': [10, 15],
        'patch_size_lr':     128,
        'patch_size_hr':     512,
        'hr_dir':            os.path.join(args.train, 'HR'),
        'lr_dir':            os.path.join(args.train, 'LR'),
        'ext':               args.train_ext,
        'batch_size':        batch_size,
        'val_hr_dir':        os.path.join(args.valid, 'HR'),
        'val_lr_dir':        os.path.join(args.valid, 'LR'),
        'val_ext':           args.valid_ext,
        'val_batch_size':    1,
        'exp_dir':           './exp/',
    }

    train(**params)


if __name__ == '__main__':
    main()
