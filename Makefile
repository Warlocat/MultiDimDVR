CPP = icpc
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
EIGEN = ~/apps/Eigen3
TEST = main.o DVR.o DVR_C.o

DVR_md.exe: ${TEST}
	${CPP} ${CPPFLAG} -I ${EIGEN} ${TEST} -o DVR_md.exe

fcf.exe: fcf.cpp
	${CPP} ${CPPFLAG} -I ${EIGEN} fcf.cpp -o fcf.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}


clean:
	rm *.o *.exe
