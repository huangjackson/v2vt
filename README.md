<!-- README.md template from https://github.com/othneildrew/Best-README-Template-->


> [!WARNING]
> Please note that this project is currently under active development and is not yet operational. Features may be incomplete, and functionality is not guaranteed.


<!-- PROJECT SHIELDS -->
[![Colab][colab-shield]][colab-url]
[![MIT License][license-shield]][license-url]
[![Hugging Face][huggingface-shield]][huggingface-url]
[![Issues][issues-shield]][issues-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
<h1 align="center">v2vt</h1>
  <p align="center">
    Video to video translation and dubbing via few shot voice cloning & audio-based lip sync.
    <br />
    <a href="https://huggingface.co/huangjackson"><strong>See the demo »</strong></a>
    <br />
    <br />
    <a href="https://github.com/huangjackson/v2vt/issues">Report Bug</a>
    ·
    <a href="https://github.com/huangjackson/v2vt/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#features">Features</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## Features

demo.mp4

Currently supports English and Chinese

* **Vocal isolation**: Isolation of vocals from source video using deep neural networks
* **Transcription**: Transcription of source video via whisper
* **Translation**: Translation from source video via CTranslate2 and OPUS-MT
* **Few-shot voice cloning**: Realistic voice cloning and TTS with as little as 5 seconds of audio from source video
* **Audio-based lip sync**: Alter faces in source video to match translated audio


<!-- GETTING STARTED -->
## Getting Started

Currently only tested in a Windows 11 environment with Python 3.9, PyTorch 2.1.1, CUDA 11.8.

### Prerequisites

* Python 3.9
* [Anaconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) (recommended)

### Manual Installation

1. Clone the repo
    ```sh
    git clone https://github.com/huangjackson/v2vt.git
    cd v2vt
    ```
2. Create a conda environment (recommended)
    ```sh
    conda create -n v2vt python=3.9
    conda activate v2vt
    ```
3. Install ffmpeg
    ```sh
    conda install ffmpeg
    ```
4. Install PyTorch and CUDA
    ```sh
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
5. Install requirements from requirements.txt
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Navigate to directory
    ```sh
    cd v2vt
    ```
2. Run CLI
    ```sh
    python v2vt.py --help
    ```


<!-- ROADMAP -->
## Roadmap

- [x] Vocal isolation
- [x] Transcription
- [x] Translation
- [x] Voice cloning/TTS
  - [ ] *Match speed of original video ([#3](https://github.com/huangjackson/v2vt/issues/3))
  - [ ] Multiple GPUs support
  - [ ] Support training & using multiple models
- [x] Lip sync
  - [ ] *Support lip sync where face isn't always present in video ([#1](https://github.com/huangjackson/v2vt/issues/1))
  - [ ] *Better face detection ([#2](https://github.com/huangjackson/v2vt/issues/2))
  - [ ] Improve inference speed
- [ ] Additional languages (currently only en & zh)
- [ ] Improve overall speed
- [ ] Improve logging ([#4](https://github.com/huangjackson/v2vt/issues/4))

See the [open issues](https://github.com/huangjackson/v2vt/issues) for a full list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Any contributions are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feat/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feat/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
See individual files and folders for any other licenses credited.


<!-- CONTACT -->
## Contact

Jackson Huang - wboh010@gmail.com

Project Link: [https://github.com/huangjackson/v2vt](https://github.com/huangjackson/v2vt)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Special thanks to the following people and projects:

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* [video-retalking](https://github.com/OpenTalker/video-retalking)
* [CTranslate2](https://github.com/OpenNMT/CTranslate2)
* [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
* [KUIELab & Woosung Choi](https://github.com/kuielab) - For the original MDX-Net music demixing model
* [KimberleyJensen](https://github.com/KimberleyJensen) - For the Kim Vocal 2 MDX-Net model
* [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) - For translation models
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)


<!-- MARKDOWN LINKS & IMAGES -->
[issues-shield]: https://img.shields.io/github/issues/huangjackson/v2vt.svg?style=for-the-badge
[issues-url]: https://github.com/huangjackson/v2vt/issues
[license-shield]: https://img.shields.io/github/license/huangjackson/v2vt.svg?style=for-the-badge
[license-url]: https://github.com/huangjackson/v2vt/blob/main/LICENSE
[colab-shield]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252
[colab-url]: https://colab.research.google.com/drive/19LCVrSCl16oVoiPnTtSJEaf1oaokgR0k?usp=sharing
[huggingface-shield]: https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=for-the-badge
[huggingface-url]: https://huggingface.co/huangjackson
