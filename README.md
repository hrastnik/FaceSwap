# [Youtube video](https://youtu.be/32i1ca8pcTg)

# How to run?

Download [OpenCV](http://opencv.org/downloads.html) and [dlib](http://dlib.net/)

- Setup OpenCV
    - Run the downloaded OpenCV executable file and select a directory to extract the OpenCV library (for example D:\opencv\)
- Setup dlib
    - Extract the downloaded zip file to a directory (for example D:\dlib)
- Download and install Visual Studio 2015 or later versions
- Run Visual Studio
- Create new empty project and name it something (for example MyProject)
- Make sure the "Debug" solution configuration is selected
- In Visual Studio open the Solution Explorer window
- Right click MyProject and choose Properties
- Click the "Configuration Manager..." button in the top left corner
- Setup the configuration for the debug
  - In the active solution platform select x64
  - Close the Configuration Manager window
  - In the property window make sure the selected Configuration in the top left is "Debug" and Platform is "x64"
  - In the panel on the left choose C/C++
  - In the Additional Include Directories field add two directories:
      - "D:\opencv\opencv\build\include"
      - "D:\dlib\dlib-19.2"
      * Note the path might be different if you have different dlib version
  - In the panel on the left choose Linker>General
  - In the Additional Library Directories add "D:\opencv\opencv\build\x64\vc14\lib"
      * Note the path might be different if you have different architecture or VS version
  - In the panel on the left choose Linker>Input
  - In the Additional Dependencies add "opencv_world320d.lib"
  - Click Apply
  
- Change the Configuration in the top left to "Release" and repeat 
- Setup the configuration for the release
  - In the panel on the left choose C/C++
  - In the Additional Include Directories field add two directories:
      - "D:\opencv\opencv\build\include"
      - "D:\dlib\dlib-19.2"
      * Note the path might be different if you have different dlib version
  - In the panel on the left choose Linker>General
  - In the Additional Library Directories add "D:\opencv\opencv\build\x64\vc14\lib"
      * Note the path might be different if you have different architecture or VS version
  - In the panel on the left choose Linker>Input
  - In the Additional Dependencies add "opencv_world320.lib"

- Close the property window
- Right click Source Files in the Solution Explorer
- Select "Add Existing Item..." and add the .cpp files from this project
- Right click Header Files in the Solution Explorer
- Select "Add Existing Item..." and add the .h files from this project
- Copy haarcascade_frontalface_default.xml from OpenCV sources/data/haarcascades directory to project directory
- Download shape_predictor_68_face_landmarks.dat from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2 and place in project directory 

After that FaceSwap should work. 

# How does it work?

The algorithm searches until it finds two faces in the frame. Then it estimates facial landmarks using dlib face landmarks. Facial landmarks are used to "cut" the faces out of the frame and to estimate the transformation matrix used to move one face over the other.

The faces are then color corrected using histogram matching and in the end the edges of the faces are feathered and blended in the original frame.

# Result
Before...

[![Before](./images/before.jpg)](https://youtu.be/32i1ca8pcTg)

After...

[![After](./images/after.jpg)](https://youtu.be/32i1ca8pcTg)
