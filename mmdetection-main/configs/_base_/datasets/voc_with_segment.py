import os


if __name__ == '__main__':
    # 生成有分割掩码的图片列表
    for i in [2007, 2012]:
        seg_dir = f'data/VOCdevkit/VOC{i}/SegmentationObject'
        mask_files = set([f.split('.')[0] for f in os.listdir(seg_dir) if f.endswith('.png')])
        with open(f'data/VOCdevkit/VOC{i}/ImageSets/Segmentation/trainval.txt', 'r') as f:
            all_imgs = [line.strip() for line in f]
        imgs_with_mask = [img for img in all_imgs if img in mask_files]
        with open(f'data/VOCdevkit/VOC{i}/ImageSets/Segmentation/trainval_with_mask.txt', 'w') as f:
            for img in imgs_with_mask:
                f.write(img + '\n')