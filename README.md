# Contrast sensitivity function in deep networks


The contrast sensitivity function (CSF) is a fundamental signature of the visual system that has been measured 
extensively in several species. It is defined by the visibility threshold for sinusoidal gratings at all spatial 
frequencies. Here, we investigated the CSF in deep neural networks using the same 2AFC contrast detection paradigm 
as in human psychophysics. Read more at: 
[bioRxiv](https://www.biorxiv.org/content/biorxiv/early/2023/04/26/2023.01.06.523034.full.pdf)
or [Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608023002186)

# Usage

## Train the linear classifier contrast discriminator

To train a linear classifier on top of a frozen pretrained network, run the command below:

```shell
python ../src/train.py \ 
-aname $MODEL --transfer_weights $MODEL_PATH $LAYER "classification" \
-dname "bw" --data_dir $DATA_DIR -b $BATCH_SIZE \
--experiment_name $EXPERIMENT_NAME -j $J --gpu 0 --output_dir $OUT_DIR \
--target_size 224 --epochs 10 --colour_space "imagenet_rgb" \
--vision_type "trichromat" --train_samples 15000 --val_sample 100 \
--contrast_space "rgb" --classifier "nn"
```
Arguments:
* `$MODEL` is the pretrained network, e.g. `resnet50`.
* `$MODEL_PATH` is the path to the weights of pretrained network, to obtain 
default weights pass the same name as `$MODEL`, e.g. `resnet50`.
* `$LAYER` is the layer to cut off the network, e.g. `fc`.
* `$DATA_DIR` the directory to the binary shape dataset, download from [here](https://www.dropbox.com/scl/fi/8si474aalbtssgpe19v09/binary_shapes.zip?rlkey=2ln2gs6dets0xm2qo6ue2qdys&dl=0).

### Measuring the network/layer CSF

After training the linear classifier, you can measure its CSF with the following command:

```shell
python ../src/test.py -aname $MODEL_PATH --contrast_space $CONTRAST_SPACE \
--experiment_name $CONTRAST_SPACE --target_size 224 --output_dir $OUT_DIR \
--colour_space "imagenet_rgb"  --vision_type "trichromat" \
--print_freq 1000 --gpu 0 --classifier "nn" --mask_image "fixed_cycle"
```

Arguments:
* `$MODEL_PATH` the path to saved *checkpoint* from the training procedure.
* `$CONTRAST_SPACE` can be one of these three strings 
`"lum_yog"`, `"rg_yog"`, `"yb_yog"` corresponding to luminance, red-green 
and yellow-blue channels.

# Citation

Akbarinia, A., Morgenstern, Y. and Gegenfurtner, K.R., 2023. Contrast sensitivity function in deep networks. 
*Neural Networks*, 164, pp.228-244.