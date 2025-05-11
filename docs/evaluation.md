# Evaluation on Mask-Benchmarks

To download the Mask-Benchmarks:

```
wget https://huggingface.co/datasets/yunjinli/Mask-Benchmark/resolve/main/Mask-Benchmark.zip
python -m zipfile -e Mask-Benchmark.zip .
```

Run the following command to compute the mIoU and mAcc respected to our benchmarks.

```
python metrics_segmentation.py -m output/<DATASET>/<NAME> --no_psnr --benchmark_path <path/to/mask-benchmark/dataset/scene>
e.g. metrics_segmentation.py -m ./sadg_example_models/sear_steak --no_psnr --benchmark_path ./Mask-Benchmark/Neu3D-Mask/sear_steak/
```
