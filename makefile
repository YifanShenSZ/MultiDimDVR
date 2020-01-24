CPP = icpc
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
EIGEN = ~/apps/Eigen3
cpplibI = ~/git_projects/cpp_lib/include
cpplibL = ~/git_projects/cpp_lib/lib
TEST = main.o DVR.o 

DVR_md.exe: ${TEST}
	${CPP} ${CPPFLAG} -I ${EIGEN} ${TEST} -L ${cpplibL} -lmkl_interface -o DVR_md.exe


%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}


clean:
	rm *.o *.exe
