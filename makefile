CPP = icpc
CPPFLAG = -O3 -std=c++11 -qopenmp
EIGEN = ~/apps/Eigen3
TEST = main.o DVR.o

DVR_md.exe: ${TEST}
	${CPP} ${CPPFLAG} -o DVR_md.exe -I ${EIGEN} ${TEST}


%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}


clean:
	rm *.o *.exe
