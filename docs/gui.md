# Tutorial on GUI

## Running the script

```
python gui_sadg.py -m output/<DATASET>/<NAME> --load_mask_on_the_fly --load_image_on_the_fly --eval --load2gpu_on_the_fly --iteration 30000
```

## Changing Different Rendering Mode

- You can use the cursor to drag to different novel views in the GUI
- Scroll to adjust the FoV
- Select different rendering modes ( `Render, Rendered Features, Gaussian Features, Gaussian Clusters, Segmentation, Point Cloud, Depth` ) in the combo box
- Please make sure to run `Clustering` before rendering the in `Gaussian Clusters, Segmentation` mode

<!-- ![demo_modes](../assets/gui_demo_modes.mp4) -->
<video autoplay loop muted playsinline>
  <source src="../assets/gui_demo_modes.mp4" type="video/mp4">
</video>

## Click Prompt

- Make sure you run `Clustering` beforehand
- Drag the slider of `Freeze Time` to change to different time
- Hold the key `A` and `Left-Click` on the object of interest in the novel view
- Hold the key `D` and `Left-Click` to deselect the click
- Drag the slider of `Score Threshold` to adjust the threshold for filtering unwanted Gaussians
-

<!-- ![demo_modes](../assets/gui_demo_click_prompt) -->

<video autoplay loop muted playsinline>
  <source src="../assets/gui_demo_click_prompt.mp4" type="video/mp4">
</video>

## Text Prompt

- Make sure you run `Clustering` beforehand
- Enter the text prompt and click `Enter`
- Click `Remove Object` to toggle the visibility of the selected objects (selected / removal)

<!-- ![demo_modes](../assets/gui_demo_text_prompt) -->
<video autoplay loop muted playsinline>
  <source src="../assets/gui_demo_text_prompt.mp4" type="video/mp4">
</video>

## Other Buttons

- You can change to different clustering methods in the combo box (DBSCAN / K-Means)
- `Render Mask` renders transparant segmentation mask on the rendering
- `Save Object` saves the current visible Gaussians (for later composition with other scenes)
- `Render Object` renders to the test camera views

## Acknowledgement

We thanks the authors from [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) for sharing their amazing work. Our GUI is built upon their code. Please consider also citing their paper.

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```
