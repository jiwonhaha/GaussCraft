# Simple Viser Viewer for 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[Editable 2D GS](https://github.com/jiwonhaha/cgvi_thesis)  <br>

This repo is editable 2D gaussian splatting by using ARAP mesh deformation.

## Installation

- If you already have the conda environment of 2D GS, then use it
- If not, follow the installation instruction from the original 2D GS

```bash
git clone https://github.com/hwanhuh/2D-GS-Viser-Viewer.git --recursive
cd 2D-GS-Viser-Viewer
pip install viser==0.1.29
pip install splines  
pip install lightning
```

## Usage
- View a 2D GS ply file 
```bash
python viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
### enable transform mode
python viewer.py <path to pre-trained model> -s <data source path> --enable_transform
```
- Train w/ viewer
```bash
python train_w_viewer.py -s <path to datas>
```
- Colab
    - You can also use the viewer in the Colab, powered by ngrok (see [example](./2dgs_viewer_colab.ipynb))
    - To use Colab and ngrok, you should add the below code to the 'start' function in the 'viewer.py' 
```python
    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        self._setup_titles(server)
        if server_config_fun is not None:
            server_config_fun(self, server)

        ### attach here!!!
        from pyngrok import ngrok
        authtoken = "your authtoken"
        ngrok.set_auth_token(authtoken)
        public_url = ngrok.connect(self.port)
        print(f"ngrok tunnel URL: {public_url}")
        ### 
```

### Control 
- **'q/e'** for up & down
- **'w/a/s/d'** for moving
- Mouse wheel for zoom in/out

## Acknowledgements
This project is built upon the following works
- [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting)
- [Viser](https://github.com/nerfstudio-project/viser)
- [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning).

Realted Blog Post: [Review of 2D Gaussian Splatting (Korean)](https://velog.io/@gjghks950/Review-2D-Gaussian-Splatting-for-Geometrically-Accurate-Radiance-Fields-Viewer-%EA%B5%AC%ED%98%84-%EC%86%8C%EA%B0%9C)
