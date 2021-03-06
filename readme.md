# Face tracking with OpenCV
Robust face tracking technique in OpenCV and dlib. Uses an optimized implementation of Normalized Cross-Correlation (NCC) to track. Runs in ~60fps on CPU.


![](assets/preview.gif)

## Getting Started (using Python virtualenv)

You need to have Python installed in your computer.

1. Install `virtualenv`:
   ```
   pip install virtualenv
   ```
2. Create a Python virtual environment:
   ```
   virtualenv venv
   ```
3. Activate virtual environment:
   1. Windows:
   ```
   cd venv\Scripts
   activate
   cd ..\..
   ```
   2. Lunix / Mac:
   ```
   source venv/bin/activate
   ```
4. Install libraries:

   ```
   pip install -r requirements.txt
   ```

## Run the code

- Run the app:
  ```
  python facetrack.py --inpath <input videos directory> --size <desired width and height (as a single number)>
  ```
