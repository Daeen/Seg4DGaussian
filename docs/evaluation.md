# Evaluation on Mask-Benchmarks

We plan to release our Mask-Benchmarks later.

Run the following command to compute the mIoU and mAcc respected to our benchmarks.

```
python metrics_segmentation.py -m output/<DATASET>/<NAME> --no_psnr --benchmark_path <path/to/mask-benchmark/dataset/scene>
```
