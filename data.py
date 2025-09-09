import random
import pathlib
import tensorflow as tf
import os 

import os
import random
import pathlib
import tensorflow as tf

# —— 1) 读取并归一化图像到 [-1,1] —— 
# def load_and_preprocess_image(image_path, ext):
#     # 读文件
#     image = tf.io.read_file(image_path)
#     # 解码
#     if ext.lower() == '.png':
#         image = tf.image.decode_png(image, channels=3)
#     else:
#         image = tf.image.decode_jpeg(image, channels=3)
#     # 转 float 并归一化到 [0,1]
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     # 归一化到 [-1,1]
#     image = image * 2.0 - 1.0
#     return image
def load_and_preprocess_image(image_path):
    """
    统一使用 decode_image 自动识别格式，
    并将像素归一化到 [-1, 1].
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(
        raw,
        channels=3,
        expand_animations=False  # 如果是 GIF，取首帧
    )
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = img * 2.0 - 1.0                                 # [-1,1]
    return img

# —— 2) 在 Dataset 流水线里把 LR/HR 双通道都解出来 —— 
def load_and_preprocess_lr_hr_images(lr_path, hr_path, ext):
    lr = load_and_preprocess_image(lr_path, ext)
    hr = load_and_preprocess_image(hr_path, ext)
    return lr, hr

# —— 3) 定义统一缩放函数 —— 
def resize_pair(lr, hr, lr_size, hr_size):
    lr = tf.image.resize(lr, [lr_size, lr_size], method='bicubic')
    hr = tf.image.resize(hr, [hr_size, hr_size], method='bicubic')
    return lr, hr

# —— 4) 根据文件夹 + 后缀，列出并打乱成 LR/HR 路径对 —— 
def get_dataset(lr_dir, hr_dir, ext):
    # 拿到所有路径
    lr_paths = sorted([str(p) for p in pathlib.Path(lr_dir).glob('*'+ext)])
    hr_paths = sorted([str(p) for p in pathlib.Path(hr_dir).glob('*'+ext)])
    # 配对并打乱顺序
    pairs = list(zip(lr_paths, hr_paths))
    random.shuffle(pairs)
    lr_paths, hr_paths = zip(*pairs)
    # 构造 Dataset
    ds = tf.data.Dataset.from_tensor_slices((list(lr_paths), list(hr_paths)))
    return ds, len(lr_paths)

# —— 5) 训练数据集：解码 → resize → shuffle → batch → repeat → prefetch —— 
def load_train_dataset(lr_dir, hr_dir, ext, batch_size, lr_size, hr_size):
    ds, n = get_dataset(lr_dir, hr_dir, ext)
    ds = ds.map(lambda l, h: load_and_preprocess_lr_hr_images(l, h, ext),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda l, h: resize_pair(l, h, lr_size, hr_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=n) \
           .batch(batch_size) \
           .repeat() \
           .prefetch(tf.data.AUTOTUNE)
    return ds, n

# —— 6) 验证数据集：解码 → resize → batch → prefetch —— 
def load_test_dataset(lr_dir, hr_dir, ext, batch_size, lr_size, hr_size):
    ds, n = get_dataset(lr_dir, hr_dir, ext)
    ds = ds.map(lambda l, h: load_and_preprocess_lr_hr_images(l, h, ext),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda l, h: resize_pair(l, h, lr_size, hr_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size) \
           .prefetch(tf.data.AUTOTUNE)
    return ds, n

# def resize_pair(lr, hr, lr_size=74, hr_size=296):
#     lr = tf.image.resize(lr, [lr_size, lr_size], method='bicubic')
#     hr = tf.image.resize(hr, [hr_size, hr_size], method='bicubic')
#     return lr, hr

# def preprocess_image(image, ext):
#     """
#     Normalize image to [-1, 1]
#     """
#     assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
#     if ext == '.png':
#         image = tf.image.decode_png(image, channels=3)
#     else:
#         image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     image -= 0.5
#     image /= 0.5

#     return image


# def load_and_preprocess_image(image_path, ext):
#     image = tf.io.read_file(image_path)
#     return preprocess_image(image, ext)


# def get_sorted_image_path(path, ext):
#     ext_regex = "*" + ext
#     data_root = pathlib.Path(path)
#     image_paths = list(data_root.glob(ext_regex))
#     image_paths = sorted([str(path) for path in image_paths])
#     return image_paths


# def get_dataset(lr_path, hr_path, ext):
#     lr_sorted_paths = get_sorted_image_path(lr_path, ext)
#     hr_sorted_paths = get_sorted_image_path(hr_path, ext)

#     lr_hr_sorted_paths = list(zip(lr_sorted_paths[:], hr_sorted_paths[:]))
#     random.shuffle(lr_hr_sorted_paths)
#     lr_sorted_paths, hr_sorted_paths = zip(*lr_hr_sorted_paths)

#     ds = tf.data.Dataset.from_tensor_slices((list(lr_sorted_paths), list(hr_sorted_paths)))

#     def load_and_preprocess_lr_hr_images(lr_path, hr_path, ext=ext):
#         return load_and_preprocess_image(lr_path, ext), load_and_preprocess_image(hr_path, ext)

#     lr_hr_ds = ds.map(load_and_preprocess_lr_hr_images, num_parallel_calls=8)
#     return lr_hr_ds, len(lr_sorted_paths)


# def load_train_dataset(lr_dir, hr_dir, ext, batch_size):
    
#     lr_hr_ds, n_data = get_dataset(lr_dir, hr_dir, ext)

#     lr_hr_ds = lr_hr_ds.shuffle(buffer_size= 200 )\
#                      .batch(batch_size)\
#                      .repeat()               # 保留 repeat，让 fit 无限迭代
#     # 不再调用 make_one_shot_iterator()
#     return lr_hr_ds, n_data


# def load_test_dataset(val_lr_dir, val_hr_dir, ext, batch_size):
#     # lr_path = os.path.join(val_lr_dir, '*'+ext)
#     # hr_path = os.path.join(val_hr_dir, '*'+ext)
#     # lr_hr_ds, n_data = get_dataset(lr_path, hr_path, ext)
#     # val_ds, n_data = get_dataset(val_lr_dir, val_hr_dir, ext)

#     # lr_hr_ds = lr_hr_ds.batch(batch_size)     # 测试时不需要 repeat
#     # # 同样删掉 make_one_shot_iterator()
#     # return lr_hr_ds, n_data
#     ds, n_data = get_dataset(val_lr_dir, val_hr_dir, ext)

#     ds = ds.batch(batch_size)     # 测试时不需要 repeat
#     return ds, n_data