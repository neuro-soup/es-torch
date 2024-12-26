{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    go_1_23
    gopls
    gotools
    gofumpt
    golangci-lint
    just
    grpcurl
    buf
    buf-language-server

    uv
    python312

    openssl

    gcc
    zlib
    glib
    stdenv.cc.cc
    python312
    uv
    clang
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    libGLU
    libGL
    fontconfig
    xorg.libXi
    xorg.libXmu
    freeglut
    freetype
    dbus
    xorg.libXext
    xorg.libX11
    xorg.libXv
    xorg.libXrandr
    xorg.libxcb
    libxkbcommon
    zlib
    ncurses5
    stdenv.cc
    binutils
    wayland
    qt5.full
    libsForQt5.qt5.qtwayland
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}:${pkgs.stdenv.cc.cc.lib}/lib/lib:${pkgs.ncurses5}/lib
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
