

<!-- PROJECT LOGO -->
<p align="center">
    <a href="https://paip2019.grand-challenge.org">
        <img src="data/images/logo.png" alt="Logo">
    </a>
    <h3 align="center">PAIP2019</h3>
    <p align="center">
        PAIP2019 is the first challenge organized by the Pathology AI Platform (PAIP)
       <br>
        <a href="https://paip2019.grand-challenge.org/"><strong>PAIP2019 homepage</strong></a>
    </p>      
</p>



<!--Table of Contents--!>

<strong>Table of contents</strong>
<details open="open">
    <ol>
        <li>
            PAIP2019 Challenge
            <ul>
                <li>background</li>
                <li>dataset</li>
                <li>evaluation</li>
            </ul>
        </li>
        <li>
            Our method
            <ul>
                <li>
                <strong>Task 1: Liver Cancer Segmentation</strong>
                </li>
                <li>
                <strong>Task 2: Viable Tumor Burden Estimation</strong>
                </li>
            </ul>
        </li>
        <li>
            How to use
        </li>

        <li>
            Results
        </li>
        <li>
            prerequisites
        </li>

    </ol>
</details>



<!--PAIP2019 challenge-->
## PAIP2019 Challenge
<ul>
    <li>
        <strong>background:</strong>
        <p>
        </p>
    </li>
    <li>
        <strong>dataset</strong></li>
        <p>
           <ul>
                <li>The training dataset contains 50 WSIs</li>
                <li>The validation dataset contains 10 WSIs</li>
                <li>The test dataset contains 40 WSIs</li>
                <p>
                    All WSIs were scanned at 40X magnification
                </p>
           </ul>
        </p>
    <li>
        <strong>evaluation</strong>
        <ul>
            <li>
               <strong>Task1: Cancer segmentation</strong>
            </li>
            <li>
                <strong>Task2: Viable Tumor Burden Estimation</strong>
            </li>
        </ul>
    </li>


</ul>

<!-- Our method -->
## Our method
<p align="center">


<!-- How to use -->
## How to use
Only support single GPU, it will take 10 minutes per each WSI.
<p align="center">
  <ol>
    <li> put your data into input_path(./dataset) (only support .svs format)
    <li> Download pretrained models into "./pretrained"
      <p>
        <a href="https://drive.google.com/drive/folders/1_XYv5gVB0OoAK8-GU5A64cVMQmkdd0He?usp=sharing">Pretrained model Link</a>
      </p>
    <li> Run run_infer.sh
    <li> you can see your result at output_path(./output).
      <p>
        <ul>
          <li> "?_v.tif" : viable tumor prediction
          <li>"?_wt.tif" : whole tumor area prediction
        </ul>
      </p>
  </ol>
</p>

<!--pretrained Weight-->
## pretrained Weight


<!-- Results -->
## Results
<ul>
    <li>
        <strong>Validation Results</strong>
        <ul>    
            <li>28 Aug. 2019</li>
            <li>Task1 score:0.6975</li>
            <li>Task2 score: 0.6558</li>
        </ul>
    </li>
    <li>
        <strong>Test Results</strong>
        <ul>
            <li>Task1 rank: 5th</li>
            <li>Task1 score: 0.665227214</li>
            <li>Task2 rank: 3rd</li>
            <li>Task2 score :0.633028622</li>
        </ul>
    </li>
</ul>



<!--Presentation-->
## Additional Resources
<ul>
    <li>
        <a href="https://drive.google.com/file/d/12ReGYi7UZF7lPau33RnMqzxeMJWCdfJj/view">Our method PPT</a>
    </li>
    <li>
        <a href="https://www.sciencedirect.com/science/article/pii/S1361841520302188">Paper of Challenge</a>
    </li>
</ul>

<!--prerequisites-->
## prerequisites
* docopt
  ```sh
  pip install docopt
  ```
* imgaug
  ```sh
  pip install imgaug
  ```
* openslide-python
  ```sh
  pip install openslide-python
  ```
* tifffile
  ```sh
  pip install tifffile
  ```
