## Initial Setting

conda 세팅법 

# 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# Miniconda 설치 스크립트 다운로드 (최신 버전)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 실행 권한 부여
chmod +x Miniconda3-latest-Linux-x86_64.sh

# 설치 실행
./Miniconda3-latest-Linux-x86_64.sh

#재부팅

source ~/miniconda3/bin/activate
conda init bash

#conda 가상환경 생성

conda create -n brain_mri python=3.10 -y(brain_mri => 원하는 이름으로 변경 가능)
conda activate brain_mri

# Github 레포지토리 클론

# 예시: brain_mri 레포지토리
git clone https://github.com/idec-gil/brain_mri.git

# 깃허브에서 토큰 발급

settings => developer settings => personal access tokens => Fine-grained tokens => generate new token

token name = ID,  repository access = all repository 체크 , permissons에서 add permission 클릭후 content 선택하고 read and write 허용 후 토큰 발급



cd brain_mri


## Environment

- Python 3.10
- CUDA 11.8
- [PyTorch](https://pytorch.org/) 2.1.0+cu118
- [Torchvision](https://pytorch.org/vision/stable/index.html) 0.16.0+cu118
- [NumPy](https://numpy.org/) 1.26.4
- [Matplotlib](https://matplotlib.org/) 3.10.5
- [tqdm](https://tqdm.github.io/) 4.67.1

### Recreate Environment

```bash
conda create -n brain_mri python=3.10 -y
conda activate brain_mri
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" "matplotlib==3.10.5" "tqdm==4.67.1"

