FROM debian:12

# Configure l'environnement pour une installation sans interaction
# DEBIAN_FRONTEND=noninteractive empêche les fenêtres de dialogue pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8  

# Mise à jour du système et installation des dépendances essentielles, inspiration des dockerfiles présent dans les dossiers des prerequis
RUN apt-get update && apt-get upgrade -y sudo && \
    apt install -y \
    metis \
    libmetis5 \
    libmetis-dev \
    apt-utils \
    aspell \
    aspell-fr \
    bash-completion \
    bison \
    ccache \
    clang-13 \
    clang-format-13 \
    cmake \
    curl \
    flex \
    g++ \
    gcc \
    gdb \
    gettext \
    gfortran \
    git \
    graphviz \
    imagemagick \
    less \
    libboost-chrono-dev \
    libboost-date-time-dev \
    libboost-filesystem-dev \
    libboost-python-dev \
    libboost-regex-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libglu1-mesa \
    libncurses5 \
    libomp-13-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libunwind-dev \
    libxcursor1 \
    libxml2-dev \
    lmod \
    locales \
    mercurial \
    nano \
    net-tools \
    openmpi-bin \
    pkg-config \
    procps \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-sphinx \
    python3-texext \
    python3-venv \
    python3-yaml \
    swig \
    tk \
    valgrind \
    vim \
    wget \
    zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/tmp/*  

# Ajoute le dépôt "sid" (version instable de Debian) pour accéder aux dernières versions de GCC et FORTRAN 
RUN echo deb http://ftp.de.debian.org/debian sid main > /etc/apt/sources.list.d/sid.list

# Crée un fichier de pinning pour éviter que les paquets sid ne remplacent d'autres paquets
RUN echo "Package: *Pin: release o=Debian,a=sid\nPin-Priority: 100" > /etc/apt/preferences.d/sid

# Mise à jour des paquets et installation GCC et FORTRAN 13
RUN apt-get update && \
    apt-get install -t sid -y gcc-13 gfortran-13 cpp-13 g++-13  \
    clang-13 clang-format-13 libomp-13-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 40 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-13 40 && \
    rm /usr/bin/cpp && rm /etc/alternatives/cpp && \
    update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-13 40 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 40

# Définit une variable d'environnement pour le répertoire de travail
ENV WRKDIR=/root/codeaster-prerequisites-20240327-oss

# Crée un répertoire pour les prérequis de code_aster et ajuster les permissions
RUN mkdir -p /opt/aster && \
	chown $(id -nu):$(id -ng) /opt/aster -R

# Téléchargement et extraction du fichier tar.gz des prérequis de code_aster
WORKDIR /root
RUN wget https://www.code-aster.org/FICHIERS/prerequisites/\
codeaster-prerequisites-20240327-oss.tar.gz && \
	tar xzf codeaster-prerequisites-20240327-oss.tar.gz
WORKDIR ${WRKDIR}

# Configuration de l'environnement Python (création d'un environnement virtuel)
# Installation des dépendances Python pour le développement de code_aster
ENV PATH=/opt/venv/bin:${PATH}
RUN \
    python3 -m venv --system-site-packages /opt/venv ; \
    python3 -m pip install --upgrade pip ; \
    python3 -m pip install -r ${WRKDIR}/reqs/requirements_dev.txt ; \
    # Installation mpi4py selon la version spécifiée dans le fichier VERSION
    mpi4py_spec=$(. ${WRKDIR}/VERSION ; printf "mpi4py==${MPI4PY}") ; \
    python3 -m pip install "${mpi4py_spec}"

# Nous activons l'utilisation de la bibliothèque système libmetis-dev
RUN sed -i 's|#LMETIS     = -L\${DEST}/metis-\${METIS}/lib -lmetis|LMETIS     = -L${DEST}/metis-${METIS}/lib -lmetis |' src/mumps.sh

# Compile en gcc-12 et 13 mais le test ne fonctionne pas => test en GCC-14 sans activer cette modification
# Nous remplaçons medium -mcmodel par large car medium n'est pas supporté en ARM avec gfortran
RUN sed -i '33i sed -i "s/-mcmodel=medium/-mcmodel=small/g" src/Makefile.inc' src/miss3d.sh

# Nous ajoutons -Wno-narrowing dans le CMakeLists de INTERP_KERNEL
RUN sed -i '49i sed -i "s/ADD_DEFINITIONS(-Wsign-compare -Wconversion)/ADD_DEFINITIONS(-Wsign-compare -Wconversion -Wno-narrowing)/g" "$(pwd)/../src/INTERP_KERNEL/CMakeLists.txt"' src/medcoupling.sh
# Lance la compilation avec les paramètres définis
RUN export DISTR=debian-12
RUN make ROOT=/opt/aster ARCH=gcc13-openblas-ompi4 RESTRICTED=0 check && \
   make ROOT=/opt/aster ARCH=gcc13-openblas-ompi4 RESTRICTED=0 setup_venv && \
   make ROOT=/opt/aster ARCH=gcc13-openblas-ompi4 RESTRICTED=0
