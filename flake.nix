{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            rocmSupport = true;
            allowUnfree = true;
          };
        };
        python = pkgs.python3.withPackages (ps: [
          ps.torch
          ps.torchvision
          ps.torchaudio
          ps.numpy
          ps.datasets          
          ps.transformers
          ps.transformers
          ps.ipython
          ps.matplotlib
          ps.pillow
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.ty
            pkgs.ruff

          ];
        };
      }
    );
}
