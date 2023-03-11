cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_INF_ENGINE=ON \
      -D WITH_NGRAPH=ON \
      -D WITH_IPP=ON \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_OPENGL=ON \
      -D WITH_PTHREADS_PF=ON \
      -D WITH_QT=ON \
      -D WITH_V4L=ON \
      -D WITH_VTK=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_JAVA=OFF \
      -D BUILD_opencv_python3=OFF ..
make -j$(nproc)
sudo make install