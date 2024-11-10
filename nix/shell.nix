{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    go_1_23
    gopls
    gotools
    gofumpt
    golangci-lint
    just
    grpcurl
  ];

  shellHook = ''
    ${pkgs.just}/bin/just setup
  '';
}
