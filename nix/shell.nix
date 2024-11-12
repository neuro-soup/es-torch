{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell rec {
  packages = with pkgs; [
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
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath packages}
  '';
}
