

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
                <p>
                    a
                </p>
                </li>
                <li>
                <strong>Task 2: Viable Tumor Burden Estimation</strong>
                <p>
                  
                </p>
                </li>
            </ul>
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
               <strong>Task1: Cancer segmentation, Jaccard Index</strong>
               <p></p> 
            </li>
            <li>
                <strong>Task2: Viable Tumor Burden Estimation</strong>
                <p>
                    Each Task1 case score is used as a weight for each Task2 case score.
                    <img src="https://latex.codecogs.com/svg.latex?\;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />  

                </p>
            </li>
        </ul>
    </li>
 

</ul>

<!-- Our method -->
## Our method
<p align="center">
    <img src="data/images/overview.PNG" alt="overview">
    <h5 align="center">Figure1.Overview</h5>
</p>
<!--prerequisites-->
## prerequisites
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

