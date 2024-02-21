<!-- README.md template from https://github.com/othneildrew/Best-README-Template-->



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
    <li>
      <a href="#features">Features</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#colab">Colab</a></li>
        <li><a href="#live-demo">Live Demo</a></li>
        <li><a href="#build-locally">Build Locally</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
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

* **Few-shot voice cloning**: Realistic voice cloning and TTS with as little as 5 seconds of audio from source video (currently supports English and Chinese)
* **Transcription/translation**: Transcription and translation from source language using whisper
* **Audio-based lip sync**: Alter faces in source video to match translated audio


<!-- GETTING STARTED -->
## Getting Started

### Colab

Visit the [Colab][colab-url] and make a copy. Requires a GPU runtime.

### Live Demo

Visit the [HuggingFace demo][huggingface-url]. Note that if there are any other active users, there will be a queue (avoidable by making a copy and running it on your own runtime).

### Build Locally

Currently only tested in a Windows 11 environment with Python 3.9, PyTorch 2.1.1, CUDA 12.4.

#### Prerequisites

Note: it is recommended to install within an Anaconda environment to prevent compatibility issues (`conda create -n v2vt python=3.9` and `conda activate v2vt`)

* ffmpeg
  ```sh
  conda install ffmpeg
  ```
* rust (may not be necessary -- install if you see errors)
  https://www.rust-lang.org/learn/get-started

#### Installation

1. Install prerequisites
2. Clone the repo
   ```sh
   git clone https://github.com/huangjackson/v2vt.git
   ```
3. Install from requirements.txt
   ```sh
   pip install -r requirements.txt
   ```



<!-- USAGE -->
## Usage

Instructions included in Colab and live demo. For local installations:

#### Local Installation - CLI

1. Navigate to directory
    ```sh
    cd v2vt
    ```
2. Run CLI
    ```sh
    python cli.py --help
    ```

#### Local Installation - Web UI

1. Navigate to directory
    ```sh
    cd v2vt
    ```
2. Run Web UI server
    ```sh
    python webui.py
    ```

<!-- ROADMAP -->
## Roadmap

- [ ] Voice cloning/TTS
- [ ] Transcription/translation
- [ ] Lip sync

See the [open issues](https://github.com/huangjackson/v2vt/issues) for a full list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Any contributions are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
See individual folders or the `/LICENSE` directory for any other licenses credited.


<!-- CONTACT -->
## Contact

Jackson Huang - wboh010@gmail.com

Project Link: [https://github.com/huangjackson/v2vt](https://github.com/huangjackson/v2vt)



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Special thanks to the following projects:

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* [whisper](https://github.com/openai/whisper)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* [video-retalking](https://github.com/OpenTalker/video-retalking)
* [hifi-gan](https://github.com/jik876/hifi-gan)
* [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
* [FFmpeg](https://github.com/FFmpeg/FFmpeg)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)


<!-- MARKDOWN LINKS & IMAGES -->
[issues-shield]: https://img.shields.io/github/issues/huangjackson/v2vt.svg?style=for-the-badge
[issues-url]: https://github.com/huangjackson/v2vt/issues
[license-shield]: https://img.shields.io/github/license/huangjackson/v2vt.svg?style=for-the-badge
[license-url]: https://github.com/huangjackson/v2vt/blob/main/LICENSE.txt
[colab-shield]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252
[colab-url]: https://colab.research.google.com/drive/19LCVrSCl16oVoiPnTtSJEaf1oaokgR0k?usp=sharing
[huggingface-shield]: https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=for-the-badge
[huggingface-url]: https://huggingface.co/huangjackson
