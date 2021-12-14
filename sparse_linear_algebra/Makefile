MPICXX=mpicxx
LAPACK_LIB=/opt/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/netlib-lapack-3.6.1-x3vu6o3gstvgr6mwaa4l6i53jqrloqjc/lib64
RAPTOR_DIR=/users/bienz/raptor
METIS_DIR=/users/bienz/packages

all : cg pcg relaxation repartition repart_relax

cg : cg.cpp
	${MPICXX} -I${RAPTOR_DIR}/raptor cg.cpp -o cg -L${RAPTOR_DIR}/build/lib -lraptor -L${LAPACK_LIB} -lblas -llapack -std=c++11


pcg : pcg.cpp
	${MPICXX} -I${RAPTOR_DIR}/raptor pcg.cpp -o pcg -L${RAPTOR_DIR}/build/lib -lraptor -L${LAPACK_LIB} -lblas -llapack -std=c++11


relaxation : relaxation.cpp
	${MPICXX} -I${RAPTOR_DIR}/raptor relaxation.cpp -o relaxation -L${RAPTOR_DIR}/build/lib -lraptor -L${LAPACK_LIB} -lblas -llapack -std=c++11

repartition : repartition.cpp
	${MPICXX} -I${RAPTOR_DIR}/raptor -I${METIS_DIR}/include repartition.cpp -o repartition -L${RAPTOR_DIR}/build/lib -lraptor -L${LAPACK_LIB} -lblas -llapack -L${METIS_DIR}/lib -lparmetis -lmetis -std=c++11

repart_relax : repart_relax.cpp
	${MPICXX} -I${RAPTOR_DIR}/raptor -I${METIS_DIR}/include repart_relax.cpp -o repart_relax -L${RAPTOR_DIR}/build/lib -lraptor -L${LAPACK_LIB} -lblas -llapack -L${METIS_DIR}/lib -lparmetis -lmetis -std=c++11

clean : 
	rm cg pcg repartition relaxation
